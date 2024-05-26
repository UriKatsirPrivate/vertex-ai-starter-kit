import os
from langchain.prompts import PromptTemplate
import vertexai
from langchain_google_vertexai import VertexAI
from flask import request

from flask import Flask, render_template

app = Flask(__name__)

project_id="landing-zone-demo-341118"

def initialize_llm():
    vertexai.init(project=project_id, location="us-central1")

    return VertexAI(
        model_name='gemini-1.5-flash-001',
        max_output_tokens=8192,
        temperature=0.3,
        top_p=0.8,
        # top_k=23,
        verbose=True,
    )

@app.route('/')
def hello():
    output = """
    This is the default message. Use /query?question=" to pass your question to the LLM
    """

    return output

# http://127.0.0.1:8080/query?question=list GCE instances
@app.route('/query')
def query_llm():
    llm=initialize_llm()
    template = """
        System: You are a virtual assistant capable of generating the corresponding Google Cloud Platform (GCP) command-line interface (CLI) command based on the user's input.

        Question: the user's input is: {user_input}.
                  Please generate the corresponding GCP CLI command. Be as elaborate as possible and use as many flags as possible.
                  For every flag you use, explain its purpose. also, make sure to provide a working sample command.

        Answer:
        """

    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm

    question = request.args.get('question')

    output = chain.invoke({"user_input":question})

    # return display(Markdown(output))
    return (output)

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
