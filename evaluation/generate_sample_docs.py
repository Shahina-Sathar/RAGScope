"""Generate sample NovaDesk PDFs for RAGScope testing."""
import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

OUT_DIR = Path(__file__).parent / "sample_docs"


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _lines(content: str) -> list[str]:
    return [ln.rstrip() for ln in content.splitlines()]


def make_pdf(title: str, body: str, filename: str) -> Path:
    """Build a simple single-page-per-section PDF with wrapped text."""
    writer = PdfWriter()

    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")

    font_bold = DictionaryObject()
    font_bold[NameObject("/Type")] = NameObject("/Font")
    font_bold[NameObject("/Subtype")] = NameObject("/Type1")
    font_bold[NameObject("/BaseFont")] = NameObject("/Helvetica-Bold")

    resources = DictionaryObject()
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = font
    fonts[NameObject("/F2")] = font_bold
    resources[NameObject("/Font")] = fonts

    page = writer.add_blank_page(width=612, height=792)

    # Build a content stream: title + wrapped body lines
    stream_parts = ["BT /F2 16 Tf 72 736 Td (%s) Tj ET" % _escape(title)]
    y = 700
    for line in _lines(body):
        if y < 60:
            # Start a new page for overflow
            page = writer.add_blank_page(width=612, height=792)
            page[NameObject("/Resources")] = resources
            y = 736
        stream_parts.append("BT /F1 11 Tf 72 %d Td (%s) Tj ET" % (y, _escape(line)))
        y -= 15

    stream = DecodedStreamObject()
    stream.set_data("\n".join(stream_parts).encode("latin-1"))
    page[NameObject("/Contents")] = stream
    page[NameObject("/Resources")] = resources

    buf = BytesIO()
    writer.write(buf)

    out_path = OUT_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(buf.getvalue())
    return out_path


DOCS = {
    "refund_policy_2025.pdf": (
        "NovaDesk Refund Policy 2025",
        """NovaDesk Software Refund Policy
Effective Date: January 1, 2025

1. Refund Period
Customers may request a refund within 14 days of the original purchase date.

2. How to Request a Refund
Submit a refund request through the customer portal at https://portal.novadesk.example.com/refunds
or contact support@novadesk.example.com.

3. Refund Processing
Approved refunds are processed within 10 business days and returned to the original
payment method.

4. Partial Refunds
Annual subscriptions cancelled after the 14 day window are not eligible for partial refunds.

5. Exclusions
Setup fees, custom integration fees, and professional services are non-refundable.""",
    ),
    "refund_policy_2026.pdf": (
        "NovaDesk Refund Policy 2026",
        """NovaDesk Software Refund Policy
Effective Date: January 1, 2026

1. Refund Period
Customers may request a refund within 30 days of the original purchase date.
This policy supersedes the 2025 refund policy.

2. How to Request a Refund
Submit a refund request through the customer portal at https://portal.novadesk.example.com/refunds
or contact support@novadesk.example.com.

3. Refund Processing
Approved refunds are processed within 5-7 business days and returned to the original
payment method.

4. Partial Refunds
Annual subscriptions cancelled after the 30 day window are not eligible for partial refunds.

5. Exclusions
Setup fees, custom integration fees, and professional services are non-refundable.""",
    ),
    "cancellation_policy.pdf": (
        "NovaDesk Cancellation Policy",
        """NovaDesk Subscription Cancellation Policy

1. Cancellation
Customers may cancel their subscription at any time from the billing settings page.

2. Effective Date
Cancellations take effect at the end of the current billing period. There is no
cancellation fee and no prorated refund for the current period.

3. Data Access
After cancellation, data remains accessible in read-only mode for 30 days.

4. Reactivation
Reactivation is available within 90 days of cancellation without data loss.""",
    ),
    "pricing.pdf": (
        "NovaDesk Pricing",
        """NovaDesk Pricing Plans

Basic - $29/month
- 5 users
- 10 GB storage
- 100 requests/hour API
- Email support

Pro - $99/month
- 25 users
- 100 GB storage
- 1000 requests/hour API
- Priority support

Enterprise - Custom
- Unlimited users
- 1 TB storage
- 10000 requests/hour API
- Dedicated support manager
- SSO and SAML included

All plans include a 14 day free trial.""",
    ),
    "plans.pdf": (
        "NovaDesk Plans Overview",
        """NovaDesk Plans and Features

Basic
- Core ticketing and helpdesk features
- 5 agents
- 10 GB attachment storage
- 100 API requests per hour

Pro
- Everything in Basic, plus:
- 25 agents
- 100 GB storage
- Advanced automation and workflows
- 1000 API requests per hour

Enterprise
- Everything in Pro, plus:
- Unlimited agents
- 1 TB storage
- Dedicated support manager
- 10000 API requests per hour
- SSO, SAML, and audit logs""",
    ),
    "api_documentation.pdf": (
        "NovaDesk API Documentation",
        """NovaDesk REST API Documentation

Authentication
All API requests require an API key passed in the Authorization header as a Bearer token.

Error Codes
- AUTH-101: Invalid API key
- AUTH-102: API key expired
- ERR-404: Endpoint not found
- ERR-429: Rate limit exceeded
- ERR-500: Internal server error
- SYNC-503: Service temporarily unavailable

Endpoints
- GET /v1/tickets
- POST /v1/tickets
- GET /v1/tickets/{id}
- PATCH /v1/tickets/{id}

Rate Limiting
See the API Limits document for per-plan request limits.""",
    ),
    "api_limits.pdf": (
        "NovaDesk API Limits",
        """NovaDesk API Rate Limits

Per-plan hourly request limits:
- Basic: 100 requests/hour
- Pro: 1000 requests/hour
- Enterprise: 10000 requests/hour

Rate Limit Error
When the limit is exceeded, the API returns error code ERR-429 with a Retry-After
header indicating when the limit resets.

Maximum Payload
The maximum request payload size is 10MB. Larger payloads return error code ERR-413.""",
    ),
    "sso_setup.pdf": (
        "NovaDesk SSO Setup Guide",
        """NovaDesk Single Sign-On (SSO) Setup

Prerequisites
SSO is available on the Enterprise plan.

Steps
1. Navigate to Admin > Security > SSO Settings.
2. Select SAML 2.0 as the protocol.
3. Upload your identity provider (IdP) metadata file.
4. Configure attribute mapping for email and display name.
5. Save and test the configuration.

Troubleshooting
If SSO login fails, verify the IdP certificate is valid and the ACS URL matches
the one provided by NovaDesk.""",
    ),
    "troubleshooting.pdf": (
        "NovaDesk Troubleshooting Guide",
        """NovaDesk Common Issues and Solutions

SYNC-503: Service temporarily unavailable
Wait 5 minutes and retry. Check the NovaDesk status page at
https://status.novadesk.example.com for maintenance windows.

AUTH-101: Invalid API key
Regenerate the API key from Admin > API Keys and update your integration.

Sync failures
Ensure the integration is on the latest version and webhooks are enabled
in the Admin > Integrations settings.""",
    ),
    "support_sla.pdf": (
        "NovaDesk Support SLA",
        """NovaDesk Support Service Level Agreement

Response Times
- Basic: 24 hours response time
- Pro: 4 hours response time
- Enterprise: 1 hour response time

Resolution Times (critical issues)
- Pro: 8 hours
- Enterprise: 4 hours

Support Channels
Support is available via email, live chat, and the support portal.""",
    ),
    "escalation_policy.pdf": (
        "NovaDesk Escalation Policy",
        """NovaDesk Ticket Escalation Policy

Initial Response
Tickets receive an initial response within the SLA window for the plan.

Escalation
If a ticket has not been resolved within 48 hours of the initial response,
it is automatically escalated to the support manager.

Critical Escalation
Critical incidents (service down, data loss) are escalated immediately to
the on-call engineering team.

Escalation Portal
Customers on Pro and Enterprise plans can escalate manually via the
escalation portal at https://support.novadesk.example.com/escalate.""",
    ),
    "privacy_policy.pdf": (
        "NovaDesk Privacy Policy",
        """NovaDesk Privacy Policy

Data We Collect
We collect account information, usage data, and support communications to
provide and improve our services.

Data We Do Not Sell
NovaDesk does not sell customer data to third parties.

Data Protection
All data is encrypted in transit (TLS) and at rest (AES-256).

Your Rights
You may request a copy, correction, or deletion of your personal data by
contacting privacy@novadesk.example.com.""",
    ),
    "data_retention.pdf": (
        "NovaDesk Data Retention Policy",
        """NovaDesk Data Retention Policy

Financial Records
Financial and billing records are retained for 7 years to comply with
tax and accounting regulations.

Support Tickets
Support tickets are retained for 3 years.

Logs
System logs are retained for 90 days.

User Data
Upon account deletion, user data is permanently removed within 30 days,
unless subject to a legal hold.""",
    ),
    "security_policy.pdf": (
        "NovaDesk Security Policy",
        """NovaDesk Security Policy

Access Control
Access to production systems requires multi-factor authentication (MFA).

Encryption
Data is encrypted at rest with AES-256 and in transit with TLS 1.2 or later.

Compliance
NovaDesk is SOC 2 Type II compliant and GDPR compliant.

Security Contacts
Report security issues to security@novadesk.example.com.""",
    ),
    "backup_policy.pdf": (
        "NovaDesk Backup Policy",
        """NovaDesk Backup Policy

Schedule
- Daily backups run at 2 AM UTC.
- Weekly full backups run every Sunday at 2 AM UTC.

Retention
- Daily backups are retained for 30 days.
- Weekly backups are retained for 1 year.

Restoration
Restoration requests must be approved by the account owner and are completed
within 24 hours.

Backup Verification
Backups are tested monthly to ensure recoverability.""",
    ),
}


def main() -> None:
    created = []
    for filename, (title, body) in DOCS.items():
        path = make_pdf(title, body, filename)
        created.append(path.name)
    print(f"Generated {len(created)} PDFs in {OUT_DIR}:")
    for name in sorted(created):
        print(f"  - {name}")


if __name__ == "__main__":
    main()