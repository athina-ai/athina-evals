import subprocess

def install():
    try:
        try:
            from guardrails import Guard
        except Exception as e:
            result = subprocess.call(["pip", "install", "-U", "guardrails-ai==0.4.2"])
            if result != 0:
                print("Guardrails installation failed")

        try:
            from guardrails.hub import GibberishText
        except Exception as e:
            result = subprocess.call(["guardrails", "hub", "install", "hub://guardrails/gibberish_text"])
            if result != 0:
                print("Guardrails installation failed. Ensure have the latest version of pip installed")

        try:
            from guardrails.hub import SensitiveTopic
        except Exception as e:
            result = subprocess.call(["guardrails", "hub", "install", "hub://guardrails/sensitive_topics"])
            if result != 0:
                print("Guardrails installation failed. Ensure have the latest version of pip installed")

        try:
            from guardrails.hub import NSFWText
        except Exception as e:
            result = subprocess.call(["guardrails", "hub", "install", "hub://guardrails/nsfw_text"])
            if result != 0:
                print("Guardrails installation failed. Ensure have the latest version of pip installed")

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
    except Exception as e:
        print(f"Error: {e}")
    