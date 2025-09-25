![AdobeStock_464438353](https://github.com/user-attachments/assets/05ffe20a-dba4-4903-943b-e689c3b7b38b)
# Boston House Price Prediction

**[Deployed URL](https://house-price-predictor-152k.onrender.com)**

This repository contains an end-to-end machine learning project on predicting house prices in Boston. The project covers steps from Exploratory Data Analysis (EDA), data cleaning, model building, saving the model as a pickle file, creating a front-end using HTML, and deploying it on the web using Render.

## About Dataset

### Context
The dataset is used to explore more on regression algorithms.

### Content
Each record in the database describes a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. The attributes are defined as follows (taken from the UCI Machine Learning Repository):

- **CRIM**: per capita crime rate by town
- **ZN**: proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**: proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: nitric oxides concentration (parts per 10 million)
- **RM**: average number of rooms per dwelling
- **AGE**: proportion of owner-occupied units built prior to 1940
- **DIS**: weighted distances to five Boston employment centers
- **RAD**: index of accessibility to radial highways
- **TAX**: full-value property-tax rate per $10,000
- **PTRATIO**: pupil-teacher ratio by town
- **B**: 1000(Bk − 0.63)² where Bk is the proportion of blacks by town
- **LSTAT**: percentage of lower status of the population
- **MEDV**: median value of owner-occupied homes in $1000s

For more information, visit the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing).

## Project Structure

- `data/`: Contains the dataset.
- `notebooks/`: Jupyter notebooks for EDA, data cleaning, and model building.
- `models/`: Contains the saved model pickle file.
- `templates/`: HTML templates for the front-end.
- `static/`: Static files for styling.
- `app.py`: Flask application for serving the predictions.
- `wsgi.py`: Entry point for Gunicorn.
- `requirements.txt`: List of dependencies.
- **`Android.apk`: <span style="color: red;">Android app for the project.</span>**
- `README.md`: Project documentation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Princeraj50/House-Price-Predictor.git
    cd house-price-predictor
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application:
    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Enter the input values for prediction and click on the "Predict" button.

## Deployment

The model is deployed on Render. You can access it [here](https://house-price-predictor-152k.onrender.com).

## License

This project is licensed under the MIT License.

## Contact
For any questions, please contact [choclateyraj50@gmail.com].
Here’s a complete working example that includes:

- ✅ Three simple tools (`sum_tool`, `subtract_tool`, `multiply_tool`)
- ✅ LangGraph ReAct agent setup
- ✅ Streamlit UI
- ✅ Requirements.txt

---

## 🧾 `app.py` — Full Code

```python
# app.py

import streamlit as st
from langchain.agents import tool, create_react_agent
from langchain.llms import OpenAI
from langgraph.graph import StateGraph

# -----------------------------
# Define Tools
# -----------------------------

@tool
def sum_tool() -> str:
    """This function performs addition."""
    print("Sum function was called")
    return "Sum function was called"

@tool
def subtract_tool() -> str:
    """This function performs subtraction."""
    print("Subtract function was called")
    return "Subtract function was called"

@tool
def multiply_tool() -> str:
    """This function performs multiplication."""
    print("Multiply function was called")
    return "Multiply function was called"

# -----------------------------
# Create ReAct Agent
# -----------------------------

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [sum_tool, subtract_tool, multiply_tool]
agent = create_react_agent(llm=llm, tools=tools)

# -----------------------------
# LangGraph Node
# -----------------------------

def agent_node(state):
    user_input = state["input"]
    result = agent.invoke({"input": user_input})
    return {"output": result}

graph = StateGraph()
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
app = graph.compile()

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="LangGraph Tool Agent", layout="centered")
st.title("🧠 LangGraph Tool Agent")

user_query = st.text_input("Ask me to add, subtract, or multiply:")

if st.button("Run"):
    result = app.invoke({"input": user_query})
    st.success(result["output"])
```

---

## 📦 `requirements.txt`

```txt
streamlit
langchain
langgraph
openai
```

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

Let me know if you want to expand this with actual math logic, LangGraph memory, or multi-step chaining. I can modularize it for Lambda or add Spinnaker orchestration next.




from langchain.tools import Tool
from langchain.llms import OpenAI
from langgraph.graph import StateGraph
from typing import TypedDict, List

# --- Setup your retriever and LLM ---
retriever = my_rag.retriever  # Your existing retriever
llm = OpenAI(model="gpt-4")   # Or any other LLM you prefer

# --- Wrap retriever as a LangChain Tool ---
retriever_tool = Tool.from_function(
    name="document_retriever",
    description="Retrieves relevant documents based on a user query.",
    func=lambda query: retriever.get_relevant_documents(query)
)

# --- Define LangGraph state ---
class RAGState(TypedDict):
    query: str
    documents: List[str]
    answer: str

# --- Node: Retrieve documents ---
def retrieve_documents(state: RAGState) -> RAGState:
    query = state["query"]
    docs = retriever_tool.run(query)
    return {**state, "documents": docs}

# --- Node: Answer with LLM ---
def answer_with_llm(state: RAGState) -> RAGState:
    context = "\n".join([doc.page_content for doc in state["documents"]])
    prompt = f"Answer the question based on context:\n{context}\n\nQuestion: {state['query']}"
    answer = llm.invoke(prompt)
    return {**state, "answer": answer}

# --- Build LangGraph ---
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_documents)
graph.add_node("answer", answer_with_llm)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "answer")
graph.set_finish_point("answer")
app = graph.compile()

# --- Run the graph ---
input_state = {"query": "What is LangGraph?", "documents": [], "answer": ""}
result = app.invoke(input_state)
print(result["answer"])







To implement a Human-in-the-Loop (HITL) interrupt step in LangGraph for agentic AI workflows, you can use LangGraph’s built-in support for conditional branching and asynchronous pauses. Here's a complete example that shows how to:

- Interrupt the agent flow for human review.
- Resume execution after human input.
- Use LangGraph’s `pause` and `resume` mechanics.

---

### 🧠 LangGraph HITL Interrupt Example

```python
from langgraph.graph import StateGraph, END, State
from typing import TypedDict, Literal, Union
from langchain.llms import OpenAI

# --- Define state ---
class AgentState(TypedDict):
    query: str
    documents: list
    answer: str
    status: Literal["pending", "approved", "rejected"]

# --- LLM setup ---
llm = OpenAI(model="gpt-4")

# --- Node: Retrieve documents ---
def retrieve_documents(state: AgentState) -> AgentState:
    # Simulate retrieval
    docs = ["LangGraph is a library for building stateful agents."]
    return {**state, "documents": docs}

# --- Node: Generate answer ---
def generate_answer(state: AgentState) -> AgentState:
    context = "\n".join(state["documents"])
    prompt = f"Answer based on context:\n{context}\n\nQuestion: {state['query']}"
    answer = llm.invoke(prompt)
    return {**state, "answer": answer, "status": "pending"}

# --- Node: Interrupt for HITL ---
def interrupt_for_review(state: AgentState) -> Union[str, State]:
    print(f"\n🔍 Human Review Needed:\nAnswer: {state['answer']}")
    print("Waiting for human approval...")

    # Pause execution and wait for external resume
    return State(status="pending")

# --- Node: Resume after HITL ---
def resume_after_review(state: AgentState) -> str:
    if state["status"] == "approved":
        return END
    elif state["status"] == "rejected":
        print("Answer rejected. Re-generating...")
        return "generate"
    else:
        return "interrupt"

# --- Build LangGraph ---
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_documents)
graph.add_node("generate", generate_answer)
graph.add_node("interrupt", interrupt_for_review)
graph.add_conditional_edges("interrupt", resume_after_review)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "interrupt")

app = graph.compile()

# --- Run the graph ---
initial_state = {"query": "What is LangGraph?", "documents": [], "answer": "", "status": "pending"}
interrupted = app.invoke(initial_state)

# Simulate human approval
interrupted["status"] = "approved"
final = app.invoke(interrupted)
print(f"\n✅ Final Answer: {final['answer']}")
```

---

### 🧩 Key Concepts
- `interrupt_for_review` pauses the graph and returns a `State(status="pending")`.
- You resume manually by updating the state and invoking again.
- Conditional edges route based on `status`: `"approved"` → END, `"rejected"` → regenerate.

This pattern is ideal for agentic flows where HITL is needed for compliance, safety, or subjective judgment. Want to plug this into a LangGraph ReAct loop or add Slack-based approval? I can scaffold that next.
