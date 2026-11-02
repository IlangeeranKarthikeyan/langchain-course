from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "openai:gpt-5"

@tool
def get_product_price(product: str) -> float:
    """Tool that retrieves the price of a product."""
    print(f"        >>EXECUTING TOOL: get_product_price with product: {product}")
    prices = {
        "laptop": 999.99,
        "headphones": 199.99,
        "keyboard": 49.99,
        "mouse": 29.99
    }
    return prices.get(product.lower(), 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Tool that applies a discount based on the discount tier to a price and returns the discounted price.
    Available discount tiers: 'bronze', 'silver', 'gold'."""
    print(f"        >>EXECUTING TOOL: apply_discount with price: {price} and discount_tier: {discount_tier}")
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount_percentage = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount_percentage / 100), 2)

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(model=MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("=" * 60)

    messages = [SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question)]

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}:")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # If no tool calls were made, the agent has finished its reasoning
        if not tool_calls:
            print("No tool calls made. Agent has finished reasoning.")
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool Call: {tool_name}, id: {tool_call_id} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in toolset.")
        observation = tool_to_use.invoke(tool_args)
        print(f" [Tool Result]: {observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call_id
        ))

    print("Max iterations reached. Agent did not finish reasoning.")


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of the laptop after applying a gold discount?")