from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable
import ollama

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


@traceable(run_type="tool")
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


@traceable(run_type="tool")
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

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
     return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    messages = [
        {
            "role": "system",
            "content": ("You are a helpful shopping assistant."
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
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}:")
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        # If no tool calls were made, the agent has finished its reasoning
        if not tool_calls:
            print("No tool calls made. Agent has finished reasoning.")
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        print(f"Tool Call: {tool_call}")
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"Tool Call: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in toolset.")
        observation = tool_to_use(**tool_args)
        print(f" [Tool Result]: {observation}")

        messages.append(ai_message)
        messages.append({
            "role": "tool",
            "content": str(observation)
        })
    print("Max iterations reached. Agent did not finish reasoning.")
    return None

if __name__ == "__main__":
    result = run_agent("What is the price of the laptop after applying a gold discount?")