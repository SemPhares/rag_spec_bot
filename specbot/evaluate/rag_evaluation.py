import pandas as pd
from typing import List
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from .custom_evaluation import evaluation_model
from .evaluation_typing import evaluation_input, base_evaluation_input, Result
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)

class Evaluation:

    def __init__(self,
                 evaluation_model: evaluation_model):
        self.evaluation_model = evaluation_model
        self.retrieval_evaluation = []
        self.generation_evaluation = []


    def build_test_case(self, 
                        test_input: evaluation_input) -> LLMTestCase:

        retrieval_context = [doc.page_content for doc in test_input.retrieval_context]
        test_case = LLMTestCase(
            input=test_input.input,
            actual_output=test_input.actual_output,
            retrieval_context=retrieval_context)
        self.test_case = test_case
        return self.test_case


    def evaluate_retrieval(self,
                           test_input: evaluation_input) -> List[Result]:

        test_case = self.build_test_case(test_input)

        # Evaluating Retrieval
        contextual_precision = ContextualPrecisionMetric(model=self.evaluation_model)
        contextual_recall = ContextualRecallMetric(model=self.evaluation_model)
        contextual_relevancy = ContextualRelevancyMetric(model=self.evaluation_model)

        retrieval_evaluation = evaluate(
            test_cases=[test_case],
            metrics=[contextual_precision, contextual_recall, contextual_relevancy]
        )
        self.retrieval_evaluation = retrieval_evaluation
        return self.retrieval_evaluation


    def evaluate_generation(self,
                            test_input: evaluation_input) -> List[Result]:

        test_case = self.build_test_case(test_input)

        # Evaluating Generation
        answer_relevancy = AnswerRelevancyMetric(model=self.evaluation_model)
        faithfulness = FaithfulnessMetric(model=self.evaluation_model)

        generation_evaluation = evaluate(
            test_cases=[test_case],
            metrics=[answer_relevancy, faithfulness]
        )

        self.generation_evaluation = generation_evaluation
        return self.generation_evaluation


    def evaluation(self,
                   test_input: evaluation_input,
                   mode :str = 'retrieval') :

        if mode == 'retrieval':
            return self.evaluate_retrieval(test_input)
        elif mode == 'generation':
            return self.evaluate_generation(test_input)
        elif mode == 'both':
            self.retrieval_evaluation = self.evaluate_retrieval(test_input)
            self.generation_evaluation = self.evaluate_generation(test_input)
            return self.retrieval_evaluation + self.generation_evaluation
        else:
            raise ValueError("Invalid mode. Choose either 'retrieval' or 'generation' or both.")


    def evaluation_to_dataframe(self):

        # current_state = base_evaluation_input()
        # Extracting attributes and their values into a dictionary
        # base = {attr: getattr(base_evaluation_input, attr) for attr in dir(base_evaluation_input) if not attr.startswith("__")}
        current_state = [vars(base_evaluation_input())]
        test_results = self.retrieval_evaluation + self.generation_evaluation
        # Convertissez la liste d'instances en une liste de dictionnaires
        test_results_dicts = [vars(tr) for tr in test_results]
        results = current_state + test_results_dicts

        # Convertissez la liste de dictionnaires en DataFrame
        df = pd.DataFrame(results)

        df.to_csv('specbot/evaluate/results/evaluation_results.csv', index=False, mode='a', header=True)

