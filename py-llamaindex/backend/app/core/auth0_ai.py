from auth0_ai.authorizers.types import Auth0ClientParams
from auth0_ai_llamaindex.auth0_ai import Auth0AI

from app.core.config import settings

auth0_ai = Auth0AI(
    Auth0ClientParams(
        {
            "domain": settings.AUTH0_DOMAIN,
            "client_id": settings.AUTH0_CLIENT_ID,
            "client_secret": settings.AUTH0_CLIENT_SECRET,
        }
    )
)

with_calendar_access = auth0_ai.with_token_vault(
    connection="google-oauth2",
    scopes=["openid", "https://www.googleapis.com/auth/calendar.events"],
)

with_gmail_read_access = auth0_ai.with_token_vault(
    connection="google-oauth2",
    scopes=["openid", "https://www.googleapis.com/auth/gmail.readonly"],
)

with_gmail_write_access = auth0_ai.with_token_vault(
    connection="google-oauth2",
    scopes=["openid", "https://www.googleapis.com/auth/gmail.compose"],
)

with_github_access = auth0_ai.with_token_vault(
    connection="github",
    scopes=[],
)

with_slack_access = auth0_ai.with_token_vault(
    connection="sign-in-with-slack",
    scopes=["channels:read", "groups:read"],
)

with_async_authorization = auth0_ai.with_async_authorization(
    audience=settings.SHOP_API_AUDIENCE,
    scopes=["openid", "product:buy"],
    binding_message=lambda product,
    quantity: f"Do you want to buy {quantity} {product}",
    user_id=lambda *_, **__: None,  # Will be set from credentials at runtime
    on_authorization_request="block",
)
