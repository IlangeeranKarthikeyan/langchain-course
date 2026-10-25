from typing import Mapping

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.tracers.schemas import Run

from .runnable import IRunnable


def report_model_time(run: Run) -> None:
    if run.end_time is None:
        raise RuntimeError("Model run ended without an end time")

    elapsed_seconds = (run.end_time - run.start_time).total_seconds()
    print("Model response time: {:.2f} seconds".format(elapsed_seconds))


class Summarizer(IRunnable):
    def run(self, context: Mapping[str, object]) -> None:
        prompt_template_factory = context["PromptTemplate"]
        llm = context["llm"]

        if not callable(prompt_template_factory):
            raise TypeError("'PromptTemplate' must be a callable")
        if not isinstance(llm, BaseChatModel):
            raise TypeError("'llm' must be a LangChain chat model")

        information = """
Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is an industrialist, technologist, and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire (in terms of US dollars) in June 2026; as of October 2026, Forbes estimates his net worth to be $1.1 trillion.

Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal. Musk also became an American citizen in 2002.

In 2002, Musk founded and became CEO and chief engineer of SpaceX, a space technology company. Musk joined Tesla as an angel investor in 2004 and became its CEO and product architect in 2008. In 2015, Musk co-founded OpenAI to advance artificial intelligence (AI) research, but later left; his growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired Twitter, a social networking service, and made significant changes, including rebranding it as X in 2023. His other businesses include Neuralink, a neurotechnology company that he co-founded in 2016, and the Boring Company, a tunneling company that he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if certain milestones are met, such as achieving a market capitalization of $8.5 trillion. Musk became the first US-dollar trillionaire upon the initial public offering of SpaceX in June 2026, though a SpaceX stock collapse brought him under the threshold by the next month.

Musk was the largest donor in the 2024 United States presidential election, in which he supported Republican nominee Donald Trump. After Trump was inaugurated in January 2025, Musk was an advisor to President Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk left the Trump administration in May 2025 and returned to managing his companies; shortly thereafter he had a public feud with Trump.

Musk is a supporter of global far-right politics, figures, and political parties. His political activities, statements and views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including spreading COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, white nationalist, racist, and transphobic comments. His acquisition of Twitter was controversial because, following his pledge to decrease censorship, there was an increase in hate speech and misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE and its cuts to the US Agency for International Development (USAID).
        """

        summary_prompt = prompt_template_factory(
            input_variables=["information"],
            template="Summarize the person in 2-3 sentences, based on this information:\n{information}",
        )
        facts_prompt = prompt_template_factory(
            input_variables=["information"],
            template="List two interesting facts about the person, based on this information:\n{information}",
        )

        timed_llm = llm.with_listeners(
            on_end=report_model_time,
            on_error=report_model_time,
        )
        chain = RunnableParallel(
            summary=summary_prompt | timed_llm,
            facts=facts_prompt | timed_llm,
        )
        result = chain.invoke({"information": information})

        print("Summary:")
        print(result["summary"].content)
        print("Interesting facts:")
        print(result["facts"].content)
