"""Flask 3.1.2 validation queries, verified against the pinned repository."""

QUICK_CODE = [
    (
        "request dispatch and view function invocation",
        ["src/flask/app.py"],
        "Find request dispatch implementation",
        "code",
        "quick",
    ),
    (
        "secure cookie session loading and saving",
        ["src/flask/sessions.py"],
        "Find secure cookie sessions",
        "code",
        "quick",
    ),
    (
        "render a Jinja template with context processors",
        ["src/flask/templating.py"],
        "Find template rendering",
        "code",
        "quick",
    ),
]

QUICK_DOC = [
    (
        "organize a Flask application with blueprints",
        ["docs/blueprints.rst"],
        "Find blueprint documentation",
        "doc",
        "quick",
    ),
    (
        "create and configure an application factory",
        ["docs/patterns/appfactories.rst"],
        "Find application factory guide",
        "doc",
        "quick",
    ),
]

QUICK_CROSS = [
    (
        "how Flask handles exceptions and registered error handlers",
        ["src/flask/app.py", "docs/errorhandling.rst"],
        "Find error handling code or guide",
        "cross",
        "quick",
    ),
    (
        "application and request context push pop lifecycle",
        ["src/flask/ctx.py", "docs/reqcontext.rst", "docs/appcontext.rst"],
        "Find context lifecycle code or docs",
        "cross",
        "quick",
    ),
]

STANDARD_CODE = [
    (
        "convert a view return value into a response object",
        ["src/flask/app.py", "src/flask/helpers.py"],
        "Find response conversion",
        "code",
        "standard",
    ),
    (
        "register a blueprint and deferred setup operations",
        ["src/flask/sansio/blueprints.py", "src/flask/blueprints.py"],
        "Find blueprint registration",
        "code",
        "standard",
    ),
    (
        "generate URLs inside and outside a request context",
        ["src/flask/app.py", "src/flask/helpers.py"],
        "Find URL generation",
        "code",
        "standard",
    ),
    (
        "class based views dispatch HTTP methods",
        ["src/flask/views.py"],
        "Find MethodView dispatch",
        "code",
        "standard",
    ),
    (
        "discover and load an application for the flask command",
        ["src/flask/cli.py"],
        "Find CLI application discovery",
        "code",
        "standard",
    ),
    (
        "serialize JSON responses with the default provider",
        ["src/flask/json/provider.py"],
        "Find JSON provider",
        "code",
        "standard",
    ),
    (
        "run before request preprocessors in blueprint order",
        ["src/flask/app.py"],
        "Find request preprocessing",
        "code",
        "standard",
    ),
]

STANDARD_DOC = [
    (
        "complete request handling lifecycle step by step",
        ["docs/lifecycle.rst"],
        "Find request lifecycle guide",
        "doc",
        "standard",
    ),
    (
        "security headers content security policy and cookie settings",
        ["docs/web-security.rst"],
        "Find web security guidance",
        "doc",
        "standard",
    ),
    (
        "use async await in route views",
        ["docs/async-await.rst"],
        "Find async view documentation",
        "doc",
        "standard",
    ),
    (
        "test an application with pytest fixtures and the test client",
        ["docs/testing.rst"],
        "Find testing guide",
        "doc",
        "standard",
    ),
    (
        "stream a response from a generator while keeping request context",
        ["docs/patterns/streaming.rst"],
        "Find streaming pattern",
        "doc",
        "standard",
    ),
    (
        "sans io base classes separate framework logic from the web server",
        ["src/flask/sansio/README.md"],
        "Find Sans-IO Markdown architecture note",
        "doc",
        "standard",
    ),
]


def get_queries(tier: str = "standard") -> list:
    """Return cumulative Flask queries for quick or standard runs."""
    queries = QUICK_CODE + QUICK_DOC + QUICK_CROSS
    if tier in {"standard", "full"}:
        queries += STANDARD_CODE + STANDARD_DOC
    return queries
