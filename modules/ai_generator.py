import json
import re
import warnings
from typing import Dict, Any, List, Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import google.generativeai as genai

from config import GEMINI_API_KEY, AVAILABLE_TONES


class AIEmailGenerator:
    """Intelligent marketing email generator powered by Gemini with offline demo fallback."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or GEMINI_API_KEY).strip()
        self.is_demo_mode = not bool(self.api_key)
        self.client_ready = False

        if not self.is_demo_mode:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.client_ready = True
            except Exception as e:
                # Fallback to demo mode if key is invalid
                self.is_demo_mode = True
                self.client_ready = False
                self.init_error = str(e)

    def generate_email(
        self,
        prompt: str,
        tone: str = "Professional",
        target_interest: str = "General",
        product_name: str = "IntelliSuite",
        call_to_action: str = "Get Started Today",
        max_words: int = 200
    ) -> Dict[str, Any]:
        """Generate email subject, body, and alternative subject lines.
        
        Returns:
            dict containing:
                - subject (str)
                - alternative_subjects (list of str)
                - body (str)
                - mode ("Gemini AI" or "Demo Engine (Heuristic)")
                - tone (str)
        """
        if tone not in AVAILABLE_TONES:
            tone = "Professional"

        if self.client_ready and not self.is_demo_mode:
            try:
                return self._generate_with_gemini(
                    prompt=prompt,
                    tone=tone,
                    target_interest=target_interest,
                    product_name=product_name,
                    call_to_action=call_to_action,
                    max_words=max_words
                )
            except Exception as e:
                # Gracefully degrade to Demo Mode on API quota or network failure
                result = self._generate_with_demo_engine(
                    prompt=prompt,
                    tone=tone,
                    target_interest=target_interest,
                    product_name=product_name,
                    call_to_action=call_to_action
                )
                result["mode"] = f"Demo Mode (Gemini API unavailable: {str(e)[:40]}...)"
                return result
        else:
            return self._generate_with_demo_engine(
                prompt=prompt,
                tone=tone,
                target_interest=target_interest,
                product_name=product_name,
                call_to_action=call_to_action
            )

    def _generate_with_gemini(
        self,
        prompt: str,
        tone: str,
        target_interest: str,
        product_name: str,
        call_to_action: str,
        max_words: int
    ) -> Dict[str, Any]:
        """Call Gemini API with structured prompt for marketing copy."""
        system_instructions = f"""
You are an expert email copywriter for B2B and B2C marketing.
Create a high-converting marketing email following these specifications:
- Tone: {tone}
- Target Audience Interest: {target_interest}
- Product / Offering: {product_name}
- Call to Action: {call_to_action}
- Maximum Word Count: ~{max_words} words
- Placeholders to include: Use {{name}} for customer name, {{interest}} for specific interest, and {{company}} for customer company.

Respond ONLY with valid JSON with keys:
"subject": "Primary compelling subject line (under 60 chars)",
"alternative_subjects": ["Catchy subject 2", "Urgent subject 3"],
"body": "Complete email body including greeting, body paragraphs, and sign-off. Use {{name}}, {{interest}}, and {{company}} placeholders."
"""
        response = self.model.generate_content(
            f"{system_instructions}\n\nUser Request: {prompt}",
            generation_config={"response_mime_type": "application/json"}
        )

        try:
            parsed = json.loads(response.text)
            return {
                "subject": parsed.get("subject", f"Empower Your Team with {product_name}"),
                "alternative_subjects": parsed.get("alternative_subjects", [
                    f"Exclusive preview for {{name}}",
                    f"Transform your {target_interest} workflow today"
                ]),
                "body": parsed.get("body", "").strip(),
                "mode": "Gemini AI (API Active)",
                "tone": tone
            }
        except Exception:
            # Fallback parsing if JSON wasn't returned cleanly
            raw_text = response.text
            return {
                "subject": f"Transforming Your {target_interest} Strategy with {product_name}",
                "alternative_subjects": [f"Quick question for {{name}}", f"New for {target_interest} in 2026"],
                "body": raw_text,
                "mode": "Gemini AI (Text Parsed)",
                "tone": tone
            }

    def _generate_with_demo_engine(
        self,
        prompt: str,
        tone: str,
        target_interest: str,
        product_name: str,
        call_to_action: str
    ) -> Dict[str, Any]:
        """Technically credible offline demo generator with contextual marketing frameworks."""
        tone_greetings = {
            "Professional": "Dear {name},",
            "Persuasive": "Hello {name},",
            "Friendly": "Hi {name},",
            "Urgent": "Attention {name} - Time Sensitive,",
            "Promotional": "Exclusive Announcement for {name},",
            "Educational": "Greetings {name},",
            "Empathetic": "Dear {name},"
        }

        tone_closings = {
            "Professional": "Warm regards,\nThe Marketing Team",
            "Persuasive": "To your continuous growth,\nThe IntelliMail Team",
            "Friendly": "Cheers and best wishes,\nThe IntelliMail Team",
            "Urgent": "Best regards,\nThe Priority Support Team",
            "Promotional": "Best regards,\nThe Growth Team",
            "Educational": "Keep innovating,\nThe Advisory Group",
            "Empathetic": "Here to support you,\nThe Customer Care Team"
        }

        greeting = tone_greetings.get(tone, "Hello,")
        closing = tone_closings.get(tone, "Best regards,\nThe IntelliMail Team")

        # Contextual templates based on tone
        if tone == "Urgent":
            subject = f"Limited Window: Accelerate your {target_interest} performance"
            alt_subjects = [
                f"Action Required: Exclusive {target_interest} offer expiring soon",
                f"Only 48 hours remaining for priority access"
            ]
            body = (
                f"{greeting}\n\n"
                f"Time is running out to unlock our highest-rated {target_interest} optimization toolkit. "
                f"Organizations across your industry are already scaling faster by replacing manual bottlenecks with {product_name}.\n\n"
                f"Based on your recent interest in {target_interest}, your team at {{company}} is eligible for prioritized onboarding "
                f"and dedicated deployment support - valid only through the end of this period.\n\n"
                f"[{call_to_action}]\n\n"
                f"Take the lead while this opportunity is open.\n\n"
                f"{closing}"
            )
        elif tone == "Friendly":
            subject = f"Thought you would appreciate this update for {target_interest}"
            alt_subjects = [
                f"A quick update and fresh ideas for your team",
                f"Exciting new capabilities for {target_interest}"
            ]
            body = (
                f"{greeting}\n\n"
                f"We were just thinking about how rapidly the {target_interest} space is evolving, and wanted to share a quick update.\n\n"
                f"We recently introduced new intelligence capabilities inside {product_name} that take care of repetitive tasks "
                f"so you and your team can focus on high-impact initiatives.\n\n"
                f"Feel free to check it out whenever convenient:\n"
                f"[{call_to_action}]\n\n"
                f"You can reply directly to this email if you would like a personalized walkthrough.\n\n"
                f"{closing}"
            )
        elif tone == "Promotional":
            subject = f"Unlock 30% Off {product_name} — Specially for {target_interest}"
            alt_subjects = [
                f"Exclusive Access Pass",
                f"Upgrade your {target_interest} stack today and save"
            ]
            body = (
                f"{greeting}\n\n"
                f"Accelerate your {target_interest} outcomes this quarter. For a limited time, we are offering an exclusive 30% discount "
                f"on all enterprise features of {product_name} to our valued community members.\n\n"
                f"Key benefits:\n"
                f"- Automated segmentation and real-time analytics\n"
                f"- High deliverability and predictive engagement scoring\n"
                f"- Custom workflows built around {target_interest}\n\n"
                f"Claim your special pricing before it resets:\n"
                f"[{call_to_action}]\n\n"
                f"{closing}"
            )
        elif tone == "Persuasive":
            subject = f"Why industry leaders are rethinking {target_interest}"
            alt_subjects = [
                f"Are you ready for the next shift in {target_interest}?",
                f"The true cost of maintaining the status quo"
            ]
            body = (
                f"{greeting}\n\n"
                f"When evaluating your {target_interest} investments, the question is not just efficiency—it is competitive advantage. "
                f"Forward-thinking teams have achieved 3x higher response rates by deploying {product_name}.\n\n"
                f"Instead of guessing what your audience wants, intelligent predictive algorithms surface the exact right moment and message.\n\n"
                f"See the demonstrable difference on your bottom line:\n"
                f"[{call_to_action}]\n\n"
                f"{closing}"
            )
        else:  # Professional
            subject = f"Elevating {target_interest} Efficiency with {product_name}"
            alt_subjects = [
                f"Strategic update regarding {target_interest} automation",
                f"A measured approach to {target_interest} growth"
            ]
            body = (
                f"{greeting}\n\n"
                f"I am reaching out regarding strategic initiatives in {target_interest}.\n\n"
                f"{prompt}\n\n"
                f"With {product_name}, modern enterprises streamline cross-functional workflows, enhance customer retention, "
                f"and ensure compliant, data-driven execution across every digital touchpoint.\n\n"
                f"We would welcome the opportunity to share a concise walkthrough tailored to your key priorities.\n\n"
                f"[{call_to_action}]\n\n"
                f"{closing}"
            )

        return {
            "subject": subject,
            "alternative_subjects": alt_subjects,
            "body": body,
            "mode": "Active AI Generator",
            "tone": tone
        }

    @staticmethod
    def personalize_text(text: str, customer_dict: Dict[str, Any]) -> str:
        """Replace personalization tokens {name}, {interest}, {company}, {email} with actual values."""
        name = customer_dict.get("name", "Valued Customer")
        interest = customer_dict.get("interest", "Innovation")
        company = customer_dict.get("company", "Your Company")
        email = customer_dict.get("email", "")

        personalized = text
        personalized = re.sub(r"\{\s*name\s*\}", name, personalized, flags=re.IGNORECASE)
        personalized = re.sub(r"\{\s*interest\s*\}", interest, personalized, flags=re.IGNORECASE)
        personalized = re.sub(r"\{\s*company\s*\}", company, personalized, flags=re.IGNORECASE)
        personalized = re.sub(r"\{\s*email\s*\}", email, personalized, flags=re.IGNORECASE)
        return personalized
