import flwr as fl
import json
import os
import sys
sys.path.append(os.getcwd() + '/../models')
sys.path.append(os.getcwd() + '/../utils')
sys.path.append(os.getcwd() + '/adversarial-robustness-toolbox')
from fs import delete_folder, create_folder, create_file, truncate


CONFIG_FILE_PATH = "../config.fl.json"
with open(CONFIG_FILE_PATH) as json_data_file:
    configs = json.load(json_data_file)

#network configs
port = configs["server"]["port"]

model_name = configs["client"]["model"]["name"]
n_epochs = configs["client"]["model"]["params"]["epochs"]
attack_name = configs["client"]["model"]["adversarial"]["attack"]
dataset = configs["data"]["name"]
num_clients = configs["fl"]["num_clients"]
num_rounds = configs["fl"]["num_rounds"]
min_eval_clients=configs["fl"]["strategy"]['min_eval_clients']


class CustomFedAvgStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
            self,
            rnd: int,
            results,
            failures,
    ):	
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None
        if not attack_name:
            experiment_results_dir = f"results/accuracies/{dataset}-{model_name}-{num_clients}clients-{n_epochs}epochs"
        else:
             experiment_results_dir = f"results/accuracies/{dataset}-adv-{attack_name}-{model_name}-{num_clients}clients-{n_epochs}epochs"
        # Create accuracies files
        raw_accuracies_filename= f"{experiment_results_dir}/raw_acc.txt"
        accuracies_filename= f"{experiment_results_dir}/acc.txt"
        if(rnd ==1):
            # Delete directory if it exists
            if os.path.exists(experiment_results_dir):
                try:
                    print("Directory exists: proceeding to delete it")
                    delete_folder(experiment_results_dir)
                except:
                    print(f"ERROR: could not delete directory {experiment_results_dir}")
            create_folder(experiment_results_dir)
            create_file(accuracies_filename)
            create_file(raw_accuracies_filename)
            
        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] *
                      r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"\n-- round {rnd} aggr accuracy: {aggregated_accuracy}\n")
        str_aggr_accuracy = truncate(aggregated_accuracy*100,2)
        if (aggregated_accuracy is not None and rnd >= 1):
            with open(raw_accuracies_filename, 'a') as file:
                file.write(f"{str_aggr_accuracy},")
            with open(accuracies_filename, 'a') as file:
                file.write(f"{rnd}:{str_aggr_accuracy}\n")
        print(f"ROUND {rnd} ACCURACY: {str_aggr_accuracy}")
        return super().aggregate_evaluate(rnd, results, failures)
    
    def aggregate_fit(
            self,
            rnd: int,
            results,
            failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        res = fl.common.parameters_to_weights(aggregated_weights[0])
        if(aggregated_weights is not None and  rnd > 0 and rnd%10==0):
            model_name = configs["client"]["model"]["name"]
            if (model_name == "resnet"):
                model = ResNet()
            elif (model_name == "vgg15"):
                model = VGG15()
            model.compile()
            model.set_weights(res)
            if not attack_name:
                model_filename = f"{dataset}-{model_name}-{rnd}rounds-{num_clients}clients-{n_epochs}epochs.h5"
            else:
                model_filename = f"{dataset}-{model_name}-adv-{attack_name}-{rnd}rounds-{num_clients}clients-{n_epochs}epochs.h5"
            print(f"======= SAVING MODEL =======")
            model.model.save("./results/"+model_filename)
        return aggregated_weights

    
def get_on_fit_config_fn():
    """Return a function which returns training configurations."""
    def fit_config(rnd: int):
        config = {
            "round": rnd,
        }
        return config

    return fit_config

def evaluate_config(rnd: int):
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}

strategy = CustomFedAvgStrategy(
    # fraction_eval=configs['server']['fraction_eval'],
    min_eval_clients=min_eval_clients,
    min_available_clients=num_clients,
    min_fit_clients=num_clients,
    on_evaluate_config_fn=evaluate_config,
    on_fit_config_fn=get_on_fit_config_fn(),
)

config = {
    "num_rounds": num_rounds
}


def main():
    print("STARTING WITH NUMBER OF ROUNDS:", config['num_rounds'])
    fl.server.start_server(f"[::]:{port}", strategy=strategy, config=config)

if __name__ == "__main__":
    main()
