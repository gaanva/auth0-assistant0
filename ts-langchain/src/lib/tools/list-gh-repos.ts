import { TokenVaultError } from '@auth0/ai/interrupts';
import { getAccessToken } from '../auth0-ai';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

const GITHUB_MCP_URL = 'https://api.githubcopilot.com/mcp/';

// Raised when the MCP call fails for an auth-related reason (expired/invalid token, missing scope).
function isAuthError(message: string) {
  return /401|403|unauthorized|bad credentials|forbidden/i.test(message);
}

// Extracts and JSON-parses the text content GitHub's MCP tools return.
function parseToolResult(result: any) {
  const text = (result.content ?? []).find((block: any) => block.type === 'text')?.text ?? '';

  if (result.isError) {
    if (isAuthError(text)) {
      throw new TokenVaultError(
        `Authorization required to access your GitHub repositories. Please connect your GitHub account.`,
      );
    }
    throw new Error(`GitHub MCP tool call failed: ${text}`);
  }

  return JSON.parse(text);
}

export const listRepositoriesTool = tool(
  async () => {
    // Get the access token from Auth0 AI
    const accessToken = await getAccessToken();

    // MCP SDK - dynamically import to avoid module resolution issues
    const { Client } = await import('@modelcontextprotocol/sdk/client/index.js');
    const { StreamableHTTPClientTransport } = await import('@modelcontextprotocol/sdk/client/streamableHttp.js');

    const client = new Client({ name: 'assistant0-github', version: '1.0.0' });
    const transport = new StreamableHTTPClientTransport(new URL(GITHUB_MCP_URL), {
      requestInit: {
        headers: { Authorization: `Bearer ${accessToken}` },
      },
    });

    try {
      await client.connect(transport);

      const me = parseToolResult(await client.callTool({ name: 'get_me', arguments: {} }));
      const search = parseToolResult(
        await client.callTool({
          name: 'search_repositories',
          arguments: { query: `user:${me.login}`, minimal_output: false },
        }),
      );

      // Return simplified repository data to avoid overwhelming the LLM
      const simplifiedRepos = (search.items ?? []).map((repo: any) => ({
        name: repo.name,
        full_name: repo.full_name,
        description: repo.description,
        private: repo.private,
        html_url: repo.html_url,
        language: repo.language,
        stargazers_count: repo.stargazers_count,
        forks_count: repo.forks_count,
        open_issues_count: repo.open_issues_count,
        updated_at: repo.updated_at,
        created_at: repo.created_at,
      }));

      return {
        total_repositories: simplifiedRepos.length,
        repositories: simplifiedRepos,
      };
    } catch (error) {
      console.log('Error', error);

      if (error instanceof TokenVaultError) {
        throw error;
      }
      if (error instanceof Error && isAuthError(error.message)) {
        throw new TokenVaultError(
          `Authorization required to access your GitHub repositories. Please connect your GitHub account.`,
        );
      }

      throw error;
    } finally {
      await client.close();
    }
  },
  {
    name: 'list_repositories',
    description: 'List data of all repositories for the current user on GitHub',
    schema: z.object({}),
  },
);
