import yaml

config_yml_path = "C:/Users/KalpeshM2/OneDrive - CitiusTech/Documents/cookiecutter-data-science/src/model/config/config.yml"

def save_mlflow_expID(expID):
    with open(config_yml_path, 'r', encoding = "utf-8") as f:
        result = yaml.safe_load(f)

    if result["mlflowexp"] == "temp":
        with open(config_yml_path, 'w', encoding = "utf-8") as f:
            result["mlflowexp"] = expID
            dump = yaml.dump(result, default_flow_style = False, allow_unicode = True, encoding = None)
            f.write(dump)

def load_mlflow_expID():
    with open(config_yml_path, 'r', encoding = "utf-8") as f:
        result = yaml.safe_load(f)
        out = result["mlflowexp"]
    return out