# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Cents, please **do not file a public GitHub issue**. Instead, email the details to:

**matt.fellows@gmail.com**

Please include reproduction steps and any relevant context. I'll acknowledge receipt and work with you on responsible disclosure before any public discussion.

## Supported Versions

Only the latest released version is supported. Please update before reporting.

## Scope

Cents is a local CLI tool that reads from third-party financial APIs and stores data in a local SQLite database. Areas of particular interest include credential handling, SQL injection in repository code, and any path that could leak API keys.
