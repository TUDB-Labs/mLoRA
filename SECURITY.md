# Security Policy for mLoRA

## Reporting a Vulnerability

If you discover a security vulnerability in mLoRA, we request that you report it responsibly.

Please **do not publicly disclose the vulnerability** until we have had a chance to assess and address it. You can report the vulnerability through email:
  
- **Email**: Send an email to `salmaf999o@gmail.com` detailing the nature of the vulnerability, steps to reproduce it, and any potential impact.

We aim to **respond to reported vulnerabilities ASAP** and will work with you to investigate and resolve the issue as quickly as possible.

## Areas of Concern

We encourage you to report issues related to:

- **Data Poisoning Attacks**: If you notice that training data is compromised or if any unexpected model behaviors occur that may result from malicious data.
- **Pipeline Parallelism Security**: If you identify vulnerabilities in the communication between nodes when using pipeline parallelism, such as unencrypted connections or unauthorized access.
- **Container Security**: Vulnerabilities in our Docker images, such as insecure configurations or exposed SSH access.
- **Credential Management**: Issues related to insecure handling of environment variables, passwords, or API keys.
- **API Misuse**: Unauthorized usage of mLoRA as a service, overloading the API, or attempting to extract sensitive model information.

## Fixing Vulnerabilities

Once a vulnerability is confirmed, we will:

1. Work on a patch to fix the vulnerability.
2. Publish a **security advisory** on GitHub to notify users of the issue and the resolution.

Thank you for helping to improve the security of mLoRA!
