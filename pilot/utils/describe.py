import os
import time
import json

DESCRIBE_PROMPT = """You're a software developer AI assistant. Your task is to explain the functionality implemented by a particular source code file.

Given a file path and file contents, your output should contain:

* a short explanation of what the file is about;
* a list of all other files referenced (imported) from this file. note that some libraries, frameworks or libraries assume file extension and don't use it explicitly. For example, "import foo" in Python references "foo.py" without specifying the extension. In your response, use the complete file name including the implied extension;

Output the result in a JSON format with the following structure, as in this example:

Example:
{
    "summary": "Describe in detail the functionality being defind o implemented in this file. Be as detailed as possible",
    "references": [
        "some/file.py",
        "some/other/file.js"
    ],
}

Your response must be a valid JSON document, following the example format. Do not add any extra explanation or commentary outside the JSON document.
"""

def _get_describe_messages(fpath: str, content: str) -> list[dict[str, str]]:
    """
    Return a list of messages to send to the AI model to describe a file.

    Internal to this module, use `describe_file` instead.

    :param fpath: the file path
    :param content: the file content
    :return: a list of messages
    """
    return [
        {"role": "system", "content": DESCRIBE_PROMPT},
        {"role": "user", "content": f"Here's the `{fpath}` file:\n```\n{content}\n```\n"},
    ]


def openai_describe(fpath: str, content: str) -> str:
    import openai

    base_url = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions").replace("/chat/completions", "")
    client = openai.OpenAI(base_url=base_url)

    print(f"[debug] OpenAI call to describe {fpath}")
    completion = client.chat.completions.create(
        messages=_get_describe_messages(fpath, content),
        temperature=0,
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content


def anthropic_describe(fpath: str, content: str) ->  str:
    import anthropic

    client = anthropic.Anthropic(
        base_url=os.getenv('ANTHROPIC_ENDPOINT') or None,
    )
    print(f"[debug] Anthropic call to describe {fpath}")
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=0,
        system=DESCRIBE_PROMPT,
        messages=_get_describe_messages(fpath, content)[1:],
    )
    return message.content[0].text


def groq_describe(fpath: str, content: str) -> str:
    import groq

    client = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))
    print(f"[debug] Groq call to describe {fpath}")
    completion = client.chat.completions.create(
        messages=_get_describe_messages(fpath, content),
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=2047,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content

def describe_file(fpath: str, content: str) -> str:
    if os.getenv('FILTER_RELEVANT_FILES', '').lower().strip() not in ['true', '1', 'yes', 'on']:
        return ''

    if not content or not content.strip():
        return "(empty)"

    # clients to use, in order of preference
    describe_function_map = [
        ("OPENAI_API_KEY", openai_describe),
        ("ANTHROPIC_API_KEY", anthropic_describe),
        ("GROQ_API_KEY", groq_describe),
    ]
    describe_function = None
    for key_name, fn in describe_function_map:
        if os.getenv(key_name):
            describe_function = fn
            break
    if not describe_function:
        return "(unknown)"

    try:
        t0 = time.time()
        desc = json.loads(describe_function(fpath, content))
        t1 = time.time()
        print(f"Generated summary for `{fpath}` in  {t1 - t0:.1f}s")
    except Exception as err:
        print(f"Error describing {fpath}: {err}")
        desc = {"summary": "(unknown)", "references": []}

    refs = (" [References: " + ", ".join(desc["references"]) + "]") if desc.get("references") else ""
    return f"{desc['summary']}{refs}"
