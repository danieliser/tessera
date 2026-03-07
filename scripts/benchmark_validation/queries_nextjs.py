"""Next.js validation queries — hand-crafted against vercel/next.js v16.1.6.

Targets: packages/next/src/ (TypeScript) + docs/ (MDX/markdown)

Each query: (query_text, expected_files, description, category, tier)
- expected_files: basenames or partial paths that count as hits
- category: "code" | "doc" | "cross"
- tier: "quick" | "standard" | "full"

All expected_files verified against the actual v16.1.6 sparse checkout.
"""

# ── Quick tier (7 queries) ─────────────────────────────────────────────

QUICK_CODE = [
    (
        "image optimization component",
        ["image-component.tsx", "image-external.tsx", "image-optimizer.ts",
         "image.tsx"],
        "Find the Image component implementation",
        "code", "quick",
    ),
    (
        "middleware request rewriting",
        ["next-url.ts", "next-request.ts", "middleware-route-matcher.ts"],
        "Find middleware URL rewrite logic",
        "code", "quick",
    ),
    (
        "route handler HTTP methods",
        ["app-route/module.ts", "auto-implement-methods.ts"],
        "Find route handler implementation",
        "code", "quick",
    ),
]

QUICK_DOC = [
    (
        "how to configure environment variables",
        ["environment-variables.mdx", "env.mdx"],
        "Find env var configuration docs",
        "doc", "quick",
    ),
    (
        "deploying Next.js application",
        ["deploying.mdx"],
        "Find deployment documentation",
        "doc", "quick",
    ),
]

QUICK_CROSS = [
    (
        "server actions form submission",
        ["action-handler.ts", "server-action-reducer.ts", "serverActions.mdx",
         "forms-and-mutations.mdx"],
        "Find server actions code and docs",
        "cross", "quick",
    ),
    (
        "dynamic route parameters slugs",
        ["dynamic-routes.mdx", "middleware-route-matcher.ts"],
        "Find dynamic routing code and docs",
        "cross", "quick",
    ),
]

# ── Standard tier (adds 23 more → 30 total) ────────────────────────────

STANDARD_CODE = [
    (
        "webpack compiler configuration",
        ["webpack-config.ts"],
        "Find webpack build config",
        "code", "standard",
    ),
    (
        "hot module replacement client",
        ["hot-reloader-webpack.ts", "hot-reloader-turbopack.ts",
         "hmr-refresh-reducer.ts"],
        "Find HMR client implementation",
        "code", "standard",
    ),
    (
        "cookie parsing and setting",
        ["cookies.ts", "get-cookie-parser.ts", "request-cookies.ts"],
        "Find cookie handling utilities",
        "code", "standard",
    ),
    (
        "static page generation at build time",
        ["static-generation-bailout.ts", "is-static-gen-enabled.ts",
         "static-site-generation.mdx"],
        "Find static generation logic",
        "code", "standard",
    ),
    (
        "error boundary fallback component",
        ["error-boundary.tsx", "not-found.ts"],
        "Find error boundary implementation",
        "code", "standard",
    ),
    (
        "link component prefetching logic",
        ["link.tsx", "prefetch.ts"],
        "Find Link prefetch behavior",
        "code", "standard",
    ),
    (
        "API route request body parsing",
        ["api-resolver.ts", "parse-body.ts"],
        "Find API route body parser",
        "code", "standard",
    ),
]

STANDARD_DOC = [
    (
        "caching and revalidation strategies",
        ["caching-and-revalidating.mdx", "caching.mdx", "revalidatePath.mdx",
         "revalidateTag.mdx"],
        "Find caching documentation",
        "doc", "standard",
    ),
    (
        "internationalization and locales",
        ["internationalization.mdx"],
        "Find i18n docs",
        "doc", "standard",
    ),
    (
        "middleware matching paths configuration",
        ["middleware-route-matcher.ts", "middleware-config.ts"],
        "Find middleware config docs",
        "doc", "standard",
    ),
    (
        "authentication and session management",
        ["authentication.mdx"],
        "Find auth documentation",
        "doc", "standard",
    ),
    (
        "CSS modules and styling options",
        ["css.mdx", "post-css.mdx", "css-in-js.mdx"],
        "Find CSS/styling docs",
        "doc", "standard",
    ),
    (
        "TypeScript configuration and support",
        ["typescript.mdx"],
        "Find TypeScript setup docs",
        "doc", "standard",
    ),
    (
        "error handling custom error pages",
        ["error-handling.mdx", "error.mdx", "not-found.mdx", "custom-error.mdx"],
        "Find error handling docs",
        "doc", "standard",
    ),
    (
        "next.config.js configuration options",
        ["next-config-js"],
        "Find config reference docs",
        "doc", "standard",
    ),
]

STANDARD_CROSS = [
    (
        "streaming and suspense boundaries",
        ["loading.mdx", "flight-render-result.ts"],
        "Find streaming implementation and docs",
        "cross", "standard",
    ),
    (
        "parallel routes and intercepting routes",
        ["parallel-routes.mdx", "intercepting-routes.mdx"],
        "Find parallel routing code and docs",
        "cross", "standard",
    ),
    (
        "metadata and open graph configuration",
        ["metadata-and-og-images.mdx", "generate-metadata.mdx",
         "resolve-metadata.ts", "opengraph-image.mdx"],
        "Find metadata system code and docs",
        "cross", "standard",
    ),
    (
        "font optimization and loading",
        ["fonts.mdx", "font.mdx", "font-utils.ts", "next-font-loader"],
        "Find font optimization code and docs",
        "cross", "standard",
    ),
    (
        "incremental static regeneration ISR",
        ["incremental-static-regeneration.mdx", "incremental-cache"],
        "Find ISR implementation and docs",
        "cross", "standard",
    ),
    (
        "data fetching with fetch API",
        ["data-fetching", "fetch-server-response.ts"],
        "Find data fetching code and docs",
        "cross", "standard",
    ),
    (
        "client components use client directive",
        ["server-and-client-components.mdx", "flight-client-entry-plugin.ts",
         "next-flight-client-entry-loader.ts"],
        "Find client component boundary code and docs",
        "cross", "standard",
    ),
    (
        "redirect and permanent redirect",
        ["redirecting.mdx", "redirect.ts", "redirect-boundary.tsx"],
        "Find redirect implementation and docs",
        "cross", "standard",
    ),
]

# ── Full tier (adds adversarial/edge cases) ─────────────────────────────

FULL_CODE = [
    (
        "turbopack module resolution",
        ["hot-reloader-turbopack.ts", "middleware-turbopack.ts"],
        "Find turbopack resolver",
        "code", "full",
    ),
    (
        "RSC payload serialization",
        ["flight-render-result.ts", "parse-and-validate-flight-router-state.tsx",
         "create-flight-router-state-from-loader-tree.ts"],
        "Find React Server Component serialization",
        "code", "full",
    ),
    (
        "trace file generation for debugging",
        ["trace.test.ts", "upload-trace.ts"],
        "Find build tracing implementation",
        "code", "full",
    ),
]

FULL_DOC = [
    (
        "upgrading from pages router to app router",
        ["app-router-migration.mdx"],
        "Find migration guide",
        "doc", "full",
    ),
    (
        "security headers content security policy",
        ["content-security-policy.mdx"],
        "Find CSP documentation",
        "doc", "full",
    ),
]

FULL_CROSS = [
    # Natural language paraphrase
    (
        "how does Next.js decide which pages to pre-render during build",
        ["static-generation-bailout.ts", "is-static-gen-enabled.ts",
         "static-site-generation.mdx", "static-exports.mdx"],
        "Natural language query about pre-rendering",
        "cross", "full",
    ),
    # Misspelled query
    (
        "middlewear route matchng",
        ["middleware-route-matcher.ts", "middleware-config.ts"],
        "Misspelled middleware query",
        "cross", "full",
    ),
    # Multi-hop
    (
        "the cache handler used by the image optimization API",
        ["image-optimizer.ts", "incremental-cache",
         "incrementalCacheHandlerPath.mdx"],
        "Multi-hop: image → optimizer → cache",
        "cross", "full",
    ),
]


def get_queries(tier: str = "standard") -> list:
    """Return queries for the specified tier (cumulative)."""
    queries = QUICK_CODE + QUICK_DOC + QUICK_CROSS

    if tier in ("standard", "full"):
        queries += STANDARD_CODE + STANDARD_DOC + STANDARD_CROSS

    if tier == "full":
        queries += FULL_CODE + FULL_DOC + FULL_CROSS

    return queries
