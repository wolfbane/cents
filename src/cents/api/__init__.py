"""Flask API for cents."""

from flask import Flask

from .errors import register_error_handlers
from .position import position_bp
from .routes import api_bp
from .thesis import thesis_bp
from .watchlist import watchlist_bp
from .research import research_bp


def create_app(config: dict | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    if config:
        app.config.update(config)

    # Register error handlers
    register_error_handlers(app)

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix="/api/v1")
    app.register_blueprint(position_bp, url_prefix="/api/v1")
    app.register_blueprint(thesis_bp, url_prefix="/api/v1")
    app.register_blueprint(watchlist_bp, url_prefix="/api/v1")
    app.register_blueprint(research_bp, url_prefix="/api/v1")

    return app
