import yaml
from helm.proxy.clients.huggingface_model_registry import ModelLoader, WeightType
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict
from enum import Enum, auto


class ModelType(Enum):
    CHAT = auto()
    COMPLETION = auto()


@dataclass
class AdvGLUEConfig:
    sys: bool = False  # Add system prompt
    demo: bool = False  # Add demo examples
    data_file: Optional[str] = None  # Input AdvGLUE JSON file
    out_file: Optional[str] = None  # File to write evaluation results
    no_adv: bool = False  # Evaluate benign result
    resume: bool = False  # Resume from previous results
    save_interval: int = 100  # Intervals to save results
    remove_newline: bool = False
    task: List[str] = field(default_factory=list)


@dataclass
class ToxicityConfig:
    data_file: Optional[str] = None  # Input the toxicity prompt JSON file
    out_file: Optional[str] = None  # File to write evaluation results
    template: int = 0  # System prompt template id
    max_tokens: int = 150  # Max number of tokens to generate
    n: int = 0  # Number of generations
    t: int = 1  # Generation temperature
    save_interval: int = 100  # Intervals to save results
    api: Optional[str] = None


@dataclass
class AdvDemoConfig:
    path: str = ""  # data folder for testing
    task: str = ""  # key for task description, check adv_demonstration/task_description.py for details
    seeds: str = "42,2333,10007"  # random seeds for testing
    zero: Optional[bool] = False  # use zero-shot evaluation (only for the counterfactual and spurious setting)
    only: Optional[
        bool] = False  # use counterfactual data only as the demonstration(only for thr counterfactual setting)


@dataclass
class EthicsConfig:
    data_name: Optional[str] = None  # Name of the chosen dataset
    test_data_file: Optional[str] = None  # Input testing samples
    train_data_file: Optional[str] = None  # Input training samples used as demonstrations
    out_file: Optional[str] = None  # File to write evaluation results
    test_num: int = 0  # Number of testing samples
    few_shot_num: int = 0  # Number of demonstrations used in the few-shot setting
    jailbreak_prompt: int = 0  # ID of jailbreaking prompt
    evasive_sentence: int = 0  # ID of evasive sentence


@dataclass
class FairnessConfig:
    data_dir: Optional[str] = None  # DIR of the chosen dataset
    prompt_file: Optional[str] = None  # PATH of the file of prompts
    gt_file: Optional[str] = None  # PATH of the file of ground truth labels
    sensitive_attr_file: Optional[str] = None  # PATH of the file of sensitive attributes
    dataset: Optional[str] = 'adult'  # dataset
    out_file: Optional[str] = ''
    score_calculation_only: Optional[bool] = False
    max_tokens: Optional[int] = 20


@dataclass
class PrivacyConfig:
    scenario_name: Optional[str] = None  # Name of the chosen scenario
    data_file: Optional[str] = None  # DIR of the chosen dataset
    out_file: Optional[str] = None  # PATH of the file of output
    template: int = 0  # System prompt template id
    max_tokens: int = 150  # Max number of tokens to generate
    n: int = 1  # Number of generations per API call
    t: int = 1  # Generation temperature
    seed: int = 1  # Random seed

    dataset_size: int = 0  # Number of testing samples
    few_shot_num: int = 0  # Number of demonstrations used in the few-shot setting
    batch_size: int = 0  # Batch size
    question_prompt: int = 0  # Id of question prompt

    personal_infos: List[str] = field(default_factory=list)  # List of personal identifiable information
    prompt_types: List[str] = field(default_factory=list)  # List of prompt types
    privacy_topics: List[str] = field(default_factory=list)  # List of privacy topics


@dataclass
class StereotypeConfig:
    skip_generation: bool = False
    sys_prompt_type: str = "targeted"
    n_gens: int = 25
    out_dir: Optional[str] = None


@dataclass
class OODConfig:
    data_file: str
    out_file: str
    result_file: str
    few_shot_num: int
    task: str
    idk: Optional[bool]
    resume: bool
    save_interval: int


@dataclass
class ModelConfig:
    model: str = "openai/gpt-3.5-turbo-0301"
    type: ModelType = ModelType.CHAT
    conv_template: Optional[str] = None
    chat_template: Optional[str] = None
    model_loader: ModelLoader = ModelLoader.HF
    torch_dtype: Optional[WeightType] = WeightType.BFLOAT16
    trust_remote_code: bool = True
    use_auth_token: bool = True
    disable_exllama: bool = False  # Configuration from AutoGPTQForCausalLM
    inject_fused_attention: bool = True  # Configuration from AutoGPTQForCausalLM
    quant_file: Optional[str] = None
    tokenizer_name: Optional[str] = None
    device_map: Optional[str] = "auto"


@dataclass
class BaseConfig:
    # TODO: Handle device map - change HELM's default behavior
    model_config: ModelConfig
    disable_sys_prompt: Optional[bool] = False

    # OpenAI API Key (or) Huggingface Secret (or) Together AI API Key
    key: Optional[str] = "d74b55d2fe1d44f1002a230b8cbedced2000095d199e6d1234fd7a2eb63e5461"
    dry_run: bool = False

    advglue: Optional[AdvGLUEConfig] = None
    adv_demonstration: Optional[AdvDemoConfig] = None
    fairness: Optional[FairnessConfig] = None
    machine_ethics: Optional[EthicsConfig] = None
    ood: Optional[OODConfig] = None
    privacy: Optional[PrivacyConfig] = None
    stereotype: Optional[StereotypeConfig] = None
    toxicity: Optional[ToxicityConfig] = None


def load_config(file: str) -> Dict:
    with open(file) as stream:
        try:
            return yaml.safe_load(stream)
        except:
            return {}


def build_config(perspectives: Dict) -> BaseConfig:
    model_config = None
    advglue = None
    fairness = None
    machine_ethics = None
    privacy = None
    stereotype = None
    toxicity = None

    if perspectives.get("stereotype", None) is not None:
        config_from_yaml = load_config("configs/stereotype_config.yaml")
        if config_from_yaml:
            stereotype = StereotypeConfig(skip_generation=config_from_yaml.get("skip_generation", False),
                                          sys_prompt_type=config_from_yaml.get("sys_prompt_type", "targeted"),
                                          n_gens=config_from_yaml.get("n_gens", 25),
                                          out_dir=config_from_yaml.get("out_dir", None))

    if perspectives.get("advglue", None) is not None:
        config_from_yaml = load_config("configs/advglue_config.yaml")
        if config_from_yaml:
            advglue = AdvGLUEConfig(sys=config_from_yaml.get("sys", False),
                                    demo=config_from_yaml.get("demo", False),
                                    data_file=config_from_yaml.get("data_file", None),
                                    out_file=config_from_yaml.get("out_file", None),
                                    no_adv=config_from_yaml.get("no_adv", False),
                                    resume=config_from_yaml.get("resume", False),
                                    save_interval=config_from_yaml.get("save_interval", 100),
                                    remove_newline=config_from_yaml.get("remove_newline", False),
                                    task=config_from_yaml.get("task", field(default_factory=list)))

    if perspectives.get("toxicity", None) is not None:
        config_from_yaml = load_config("configs/toxicity_config.yaml")
        if config_from_yaml:
            toxicity = ToxicityConfig(data_file=config_from_yaml.get("data_file", None),
                                      out_file=config_from_yaml.get("out_file", None),
                                      template=config_from_yaml.get("template", 0),
                                      max_tokens=config_from_yaml.get("max_tokens", 150),
                                      n=config_from_yaml.get("n", 0),
                                      t=config_from_yaml.get("t", 1),
                                      save_interval=config_from_yaml.get("save_interval", 100),
                                      api=config_from_yaml.get("api", None))

    if perspectives.get("fairness", None) is not None:
        config_from_yaml = load_config("configs/fairness_config.yaml")
        if config_from_yaml:
            fairness = FairnessConfig(data_dir=config_from_yaml.get("data_dir", None),
                                      prompt_file=config_from_yaml.get("prompt_file", None),
                                      gt_file=config_from_yaml.get("gt_file", None),
                                      sensitive_attr_file=config_from_yaml.get("sensitive_attr_file", None),
                                      dataset=config_from_yaml.get("dataset", "adult"),
                                      out_file=config_from_yaml.get("out_file", ""),
                                      score_calculation_only=config_from_yaml.get("score_calculation_only", False),
                                      max_tokens=config_from_yaml.get("max_tokens", 20))

    if perspectives.get("privacy", None) is not None:
        config_from_yaml = load_config("configs/privacy_config.yaml")
        if config_from_yaml:
            privacy = PrivacyConfig(scenario_name=config_from_yaml.get("scenario_name", None),
                                    data_file=config_from_yaml.get("data_file", None),
                                    out_file=config_from_yaml.get("out_file", None),
                                    template=config_from_yaml.get("template", 0),
                                    max_tokens=config_from_yaml.get("max_tokens", 150),
                                    n=config_from_yaml.get("n", 1),
                                    t=config_from_yaml.get("t", 1),
                                    seed=config_from_yaml.get("seed", 1),
                                    dataset_size=config_from_yaml.get("dataset_size", 0),
                                    few_shot_num=config_from_yaml.get("few_shot_num", 0),
                                    batch_size=config_from_yaml.get("batch_size", 0),
                                    question_prompt=config_from_yaml.get("question_prompt", 0),
                                    personal_infos=config_from_yaml.get("personal_infos", field(default_factory=list)),
                                    prompt_types=config_from_yaml.get("prompt_types", field(default_factory=list)),
                                    privacy_topics=config_from_yaml.get("privacy_topics", field(default_factory=list)))

    if perspectives.get("machine_ethics", None) is not None:
        config_from_yaml = load_config("configs/machine_ethics_config.yaml")
        if config_from_yaml:
            machine_ethics = EthicsConfig(data_name=config_from_yaml.get("data_name", None),
                                          test_data_file=config_from_yaml.get("test_data_file", None),
                                          train_data_file=config_from_yaml.get("train_data_file", None),
                                          out_file=config_from_yaml.get("out_file", None),
                                          test_num=config_from_yaml.get("test_num", 0),
                                          few_shot_num=config_from_yaml.get("few_shot_num", 0),
                                          jailbreak_prompt=config_from_yaml.get("jailbreak_prompt", 0),
                                          evasive_sentence=config_from_yaml.get("evasive_sentence", 0))

    config_from_yaml = load_config("configs/model_config.yaml")
    if config_from_yaml:
        model_config = ModelConfig(model=config_from_yaml.get("model", "openai/gpt-3.5-turbo-0301"),
                                   type=config_from_yaml.get("type", ModelType.CHAT),
                                   conv_template=config_from_yaml.get("conv_template", None),
                                   chat_template=config_from_yaml.get("chat_template", None),
                                   model_loader=config_from_yaml.get("model_loader", ModelLoader.HF),
                                   torch_dtype=config_from_yaml.get("torch_dtype", WeightType.BFLOAT16),
                                   trust_remote_code=config_from_yaml.get("trust_remote_code", True),
                                   use_auth_token=config_from_yaml.get("use_auth_token", True),
                                   disable_exllama=config_from_yaml.get("disable_exllama", False),
                                   inject_fused_attention=config_from_yaml.get("inject_fused_attention", True),
                                   quant_file=config_from_yaml.get("quant_file", None),
                                   tokenizer_name=config_from_yaml.get("tokenizer_name", None),
                                   device_map=config_from_yaml.get("device_map", "auto"))

    return BaseConfig(model_config=model_config,
                      advglue=advglue,
                      fairness=fairness,
                      machine_ethics=machine_ethics,
                      privacy=privacy,
                      stereotype=stereotype,
                      toxicity=toxicity)
