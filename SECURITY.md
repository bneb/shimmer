# Security Policy

## Reporting a Vulnerability

Send reports via GitHub's [private vulnerability reporting](https://github.com/bneb/shimmer/security/advisories) or email the project maintainers.

We aim to acknowledge receipt within 48 hours and provide an initial assessment
within 5 business days. Critical issues will receive priority fixes.

## Scope

The following areas are in scope for vulnerability reports:

- **Tool execution subsystem**: shell command dispatch, output handling, argument
  parsing, sandbox escapes.
- **Model input parsing**: chat template rendering, XML/JSON extraction,
  interceptor buffer handling.
- **API surface**: HTTP server endpoints, socket-based daemon IPC, file system
  access paths.

Out of scope: third-party model weights, upstream LLM libraries, and the
llama.cpp runtime itself unless the vulnerability is exercised through Shimmer's
integration layer.

## Response Timeline

| Severity | Assessment | Fix Target |
|----------|------------|------------|
| Critical | 5 business days | 14 days |
| High     | 5 business days | 30 days |
| Medium   | 10 business days | 90 days |
| Low      | 10 business days | Next release |
