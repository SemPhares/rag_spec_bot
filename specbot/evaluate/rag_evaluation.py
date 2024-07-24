import uuid
import pandas as pd
from copy import deepcopy
from datetime import datetime
from deepeval import evaluate
from typing import List, Union, Literal
from deepeval.test_case import LLMTestCase
from .custom_evaluation import evaluation_model
from .evaluation_typing import evaluation_input, base_evaluation_input, Result
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric)


class Evaluation:

    def __init__(self,
                 test_input: Union[evaluation_input, List[evaluation_input]],
                 evaluation_model: evaluation_model):
        
        self.eval_results = []
        self.test_input = test_input
        self.evaluation_model = evaluation_model
        self.test_case: List[LLMTestCase] = self.build_test_case()


    def _test_case(self, 
                   test_input:evaluation_input) -> LLMTestCase:
        
        test_case = LLMTestCase(input=test_input.input,
                    actual_output=test_input.actual_output,
                    retrieval_context=[doc.page_content for doc in test_input.retrieval_context])
        
        return test_case


    def build_test_case(self) -> List[LLMTestCase]:

        if isinstance(self.test_input, list):
            self.test_case = [self._test_case(ti) for ti in self.test_input]
            return self.test_case
        
        else:
            self.test_case = [self._test_case(self.test_input)]
            return self.test_case


    def evaluate_retrieval(self) -> List[Result]:

        # Evaluating Retrieval
        contextual_precision = ContextualPrecisionMetric(model=self.evaluation_model)
        contextual_recall = ContextualRecallMetric(model=self.evaluation_model)
        contextual_relevancy = ContextualRelevancyMetric(model=self.evaluation_model)

        retrieval_evaluation = evaluate(
            test_cases=self.test_case,
            metrics=[contextual_precision, contextual_recall, contextual_relevancy])
        
        return retrieval_evaluation


    def evaluate_generation(self) -> List[Result]:

        # Evaluating Generation
        answer_relevancy = AnswerRelevancyMetric(model=self.evaluation_model)
        faithfulness = FaithfulnessMetric(model=self.evaluation_model)

        generation_evaluation = evaluate(
            test_cases=self.test_case,
            metrics=[answer_relevancy, 
                    #  faithfulness
                     ],
            use_cache=True,
            write_cache=True,
            cache_dir='specbot/evaluate/cache')
        
        return generation_evaluation
    

    def _results_to_dict(self,
                         evalualtion_results: List[Result]) -> list[dict]:
        
        resutls = deepcopy(evalualtion_results)
        current_state = base_evaluation_input(
            CURRENT_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        test_results = [vars(tr) for tr in resutls]

        for result in test_results:
            for i, metric_meta in enumerate(result['metrics_metadata']):
                result.update({'filename': self.test_input.file_name})
                result.update({'generation_time': self.test_input.generation_time})
                for key, value in vars(metric_meta).items():
                    result.update({f'{key}_{i}': value})
            del result['metrics_metadata']

        for result in test_results :
            for key, value in vars(current_state).items():
                result.update({key: value})
        
        return test_results
                

    def _results_to_dataframe(self, 
                              evalualtion_results: List[Result]) -> pd.DataFrame:
        
        # Convertissez la liste d'instances en une liste de dictionnaires
        test_results = self._results_to_dict(evalualtion_results)

        # Convertissez la liste de dictionnaires en DataFrame
        self.df = pd.DataFrame(test_results)

        self.save_results()
        return self.df
    

    def save_results(self):
        self.df.to_csv(f'specbot/evaluate/results/results_{str(uuid.uuid4())}.csv', 
                       index=False, mode='a', header=True)


    def evaluation(self,
                   evalualtion_results: List[Result] = [],
                   evaluation_mode: Literal['retrieval', 'both', 'generation'] = 'retrieval',
                   display_mode: Literal['dict', 'dataframe', 'raw_result'] = 'dataframe'):
        
        if not evalualtion_results:

            # Evaluate the model
            if evaluation_mode == 'retrieval':
                self.eval_results = self.evaluate_retrieval()
            
            elif evaluation_mode == 'generation':
                self.eval_results = self.evaluate_generation()
            
            elif evaluation_mode == 'both':
                self.retrieval_evaluation = self.evaluate_retrieval()
                self.generation_evaluation = self.evaluate_generation()
                self.eval_results = self.retrieval_evaluation + self.generation_evaluation

            else:
                raise ValueError("Invalid mode. Choose either 'retrieval' or 'generation' or both.")

        else:
            self.eval_results = evalualtion_results


        # Display the results
        if display_mode == 'dict':
            return self._results_to_dict(self.eval_results)
        elif display_mode == 'dataframe':
            return self._results_to_dataframe(self.eval_results)
        else:
            return self.eval_results
