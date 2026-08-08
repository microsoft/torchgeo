# Security Policy

TorchGeo takes its security very seriously. If you believe you have found a vulnerability in TorchGeo, we encourage you to [report it privately](#reporting-a-vulnerability). Below we document TorchGeo's security policy.

## Supported Versions

We provide security updates for `main` and for the most recent minor (`X.Y`) release branch. Security fixes will not be backported to older versions of TorchGeo, and security-conscious users are encouraged to keep all dependencies up to date. Security updates will be made available as patch (`X.Y.1`, `X.Y.2`, etc.) releases. For more on TorchGeo's release structure, see our [Releasing Instructions](https://github.com/torchgeo/torchgeo/wiki/Releasing-Instructions).

## Issues That Are Not Security Vulnerabilities

The following examples _do not_ constitute security vulnerabilities:

- **Vulnerabilities in our dependencies.** For example, vulnerabilities in low-level dependencies like Pillow are frequently discovered and could affect TorchGeo usage. These vulnerabilities should be reported to [Pillow](https://github.com/python-pillow/Pillow/security) instead.
- **Vulnerabilities in undocumented APIs.** For example, we use `pickle.loads()` in our unit tests, but these are not user-facing and do not constitute a vulnerability.
- **Intentionally disabling or bypassing security features.** For example, the use of `checksum=False` or manually tampering with the download URL or checksum is not considered secure and is a user-mistake instead of a security vulnerability.

When in doubt, open a vulnerability report and check with us.

## Issues That Are Security Vulnerabilities

The following examples _do_ constitute security vulnerabilities:

- **Insecure usage of our dependencies.** For example, the use of `eval`[^1], `exec`, `pickle.loads`, `subprocess.run(shell=True)`, or `torch.load(weights_only=False)`[^2] are standard features of Python and PyTorch. But without input validation, these can result in remote code execution.
- **Vulnerabilities in CI.** For example, a vulnerability that allows exfiltration of private API keys would constitute a security vulnerability, not in TorchGeo, but in our release pipeline.
- **Vulnerabilities in our downloads.** For example, zip bombs or viruses in our downloaded datasets or model weights would constitute a security vulnerability and should be reported.

This list is not exhaustive, and we encourage you to report when in doubt.

[^1]: https://github.com/torchgeo/torchgeo/security/advisories/GHSA-ghq9-vc6f-8qjf

[^2]: https://github.com/torchgeo/torchgeo/security/advisories/GHSA-6gm9-8jxc-p862

## Known Security Considerations

TorchGeo uses checksums to verify the validity of all downloads. We are currently in the process of migrating cryptographically insecure algorithms like MD5 to cryptographically secure algorithms like SHA256. Users concerned about this can independently verify all downloads before decompression and usage.

## Reporting a Vulnerability

You can report a vulnerability using GitHub's private reporting feature:

1. Go to https://github.com/torchgeo/torchgeo/security/advisories/new.
2. Fill out the form to the best of your abilities.
3. Click "Create draft security advisory".

More details are available in [GitHub's documentation](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability). As documented in our [AI Policy](https://github.com/torchgeo/governance/blob/main/AI-POLICY.md), LLM usage should be disclosed, and the vulnerability report should be written by a human, not their agent. Detailed reports with proof-of-concept exploits are encouraged but not required. If a report is accepted and you do not yet have a CVE ID, we will request one on your behalf and attribute the discovery to you.

## Timeline

We typically strive for the following timeline:

1. Reported vulnerabilities will be investigated and you can expect feedback within two days.
2. Accepted vulnerabilities will be fixed within one week.
3. A new release containing the fix will be made within two weeks.
4. Vulnerabilities will be disclosed after a standard 90-day wait period to allow users time to upgrade.

If fixing the issue or releasing a fix will take longer than this, we will discuss timeline options with you.
