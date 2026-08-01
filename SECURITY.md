# Security Policy

Please report security issues privately through GitHub's security-advisory feature rather than a
public issue.

This project loads Hugging Face custom model code with `trust_remote_code=True`. Pin and review
the model revision before running it. Keep API keys in environment variables or platform secret
stores, and do not put sensitive text in W&B metadata or public datasets.
