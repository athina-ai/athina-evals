import subprocess

def main():
    print("Installing guardrails")
    try:
        from guardrails import Guard
    except Exception as e:
        subprocess.check_call(["pip", "install", "-U", "guardrails-ai"])

    try:
        from guardrails.hub import GibberishText
    except Exception as e:
        subprocess.check_call(["guardrails", "hub", "install", "hub://guardrails/gibberish_text"])

    try:
        from guardrails.hub import SensitiveTopic
    except Exception as e:
        subprocess.check_call(["guardrails", "hub", "install", "hub://guardrails/sensitive_topics"])

    try:
        from guardrails.hub import NSFWText
    except Exception as e:
        subprocess.check_call(["guardrails", "hub", "install", "hub://guardrails/nsfw_text"])

    try:
        import nltk
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        nltk.download('punkt')
    except Exception as e:
        print(f"Error: {e}")

    print("Done installing guardrails")