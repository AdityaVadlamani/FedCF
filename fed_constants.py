from enum import Enum, EnumMeta
import re

from folktables.load_acs import state_list


class CommFormulations(Enum):
    LOW_OVERHEAD = 0
    MORE_PRIVATE = 1


CLIENT_ID = "client_id"

FOLKTABLES_CONTINENTAL_SMALL = "continental_small"
FOLKTABLES_SMALL = "small"
FOLKTABLES_CONTINENTAL_LARGE = "continental_large"
FOLKTABLES_LARGE = "large"
FOLKTABLES_CONTINENTAL_ALL = "continental_all"
FOLKTABLES_ALL = "all"

FOLKTABLES_OPTIONS = [
    FOLKTABLES_CONTINENTAL_SMALL,
    FOLKTABLES_SMALL,
    FOLKTABLES_CONTINENTAL_LARGE,
    FOLKTABLES_LARGE,
    FOLKTABLES_CONTINENTAL_ALL,
    FOLKTABLES_ALL,
]

FOLKTABLES_SPLIT = {
    FOLKTABLES_ALL: {st: [st] for st in state_list},
    FOLKTABLES_CONTINENTAL_ALL: {
        st: [st] for st in state_list if st not in ("AK", "HI", "PR")
    },
    FOLKTABLES_SMALL: {
        "NORTHEAST": sorted(
            ["PA", "CT", "MA", "ME", "NH", "RI", "VT", "DE", "MD", "NJ", "NY"]
        ),
        "SOUTH": sorted(
            [
                "TX",
                "OK",
                "AR",
                "LA",
                "MS",
                "TN",
                "KY",
                "AL",
                "GA",
                "SC",
                "NC",
                "VA",
                "WV",
                "FL",
            ]
        )
        + ["PR"],
        "WEST": sorted(
            [
                "ID",
                "UT",
                "MT",
                "WY",
                "CO",
                "WA",
                "OR",
                "CA",
                "NV",
                "AZ",
                "NM",
                "AK",
                "HI",
            ]
        ),
        "MIDWEST": sorted(
            [
                "IN",
                "IL",
                "MI",
                "OH",
                "WI",
                "ND",
                "SD",
                "NE",
                "KS",
                "MN",
                "IA",
                "MO",
            ]
        ),
    },
    FOLKTABLES_CONTINENTAL_SMALL: {
        "NORTHEAST": sorted(
            ["PA", "CT", "MA", "ME", "NH", "RI", "VT", "DE", "MD", "NJ", "NY"]
        ),
        "SOUTH": sorted(
            [
                "TX",
                "OK",
                "AR",
                "LA",
                "MS",
                "TN",
                "KY",
                "AL",
                "GA",
                "SC",
                "NC",
                "VA",
                "WV",
                "FL",
            ]
        ),
        "WEST": sorted(
            [
                "ID",
                "UT",
                "MT",
                "WY",
                "CO",
                "WA",
                "OR",
                "CA",
                "NV",
                "AZ",
                "NM",
            ]
        ),
        "MIDWEST": sorted(
            [
                "IN",
                "IL",
                "MI",
                "OH",
                "WI",
                "ND",
                "SD",
                "NE",
                "KS",
                "MN",
                "IA",
                "MO",
            ]
        ),
    },
    FOLKTABLES_LARGE: {
        "NEW_ENGLAND": sorted(["CT", "MA", "ME", "NH", "RI", "VT"]),
        "MIDEAST": sorted(["DE", "MD", "NJ", "NY", "PA"]),
        "GREAT_LAKES": sorted(["IN", "IL", "MI", "OH", "WI"]),
        "SOUTHEAST": sorted(
            [
                "AR",
                "LA",
                "MS",
                "TN",
                "KY",
                "AL",
                "GA",
                "SC",
                "NC",
                "VA",
                "WV",
                "FL",
            ]
        )
        + ["PR"],
        "PLAINS": sorted(["ND", "SD", "NE", "KS", "MN", "IA", "MO"]),
        "SOUTHWEST": sorted(["AZ", "NM", "TX", "OK"]),
        "ROCKY_MOUNTAINS": sorted(["ID", "UT", "MT", "WY", "CO"]),
        "FAR_WEST": sorted(["WA", "OR", "CA", "NV", "AK", "HI"]),
    },
    FOLKTABLES_CONTINENTAL_LARGE: {
        "NEW_ENGLAND": sorted(["CT", "MA", "ME", "NH", "RI", "VT"]),
        "MIDEAST": sorted(["DE", "MD", "NJ", "NY", "PA"]),
        "GREAT_LAKES": sorted(["IN", "IL", "MI", "OH", "WI"]),
        "SOUTHEAST": sorted(
            [
                "AR",
                "LA",
                "MS",
                "TN",
                "KY",
                "AL",
                "GA",
                "SC",
                "NC",
                "VA",
                "WV",
                "FL",
            ]
        ),
        "PLAINS": sorted(["ND", "SD", "NE", "KS", "MN", "IA", "MO"]),
        "SOUTHWEST": sorted(["AZ", "NM", "TX", "OK"]),
        "ROCKY_MOUNTAINS": sorted(["ID", "UT", "MT", "WY", "CO"]),
        "FAR_WEST": sorted(["WA", "OR", "CA", "NV"]),
    },
}

PARTITION_ID_TO_KEY = {
    FOLKTABLES_ALL: state_list,
    FOLKTABLES_CONTINENTAL_ALL: [
        state for state in state_list if state not in ("AK", "HI", "PR")
    ],
    FOLKTABLES_SMALL: list(sorted(FOLKTABLES_SPLIT[FOLKTABLES_SMALL].keys())),
    FOLKTABLES_LARGE: list(sorted(FOLKTABLES_SPLIT[FOLKTABLES_LARGE].keys())),
    FOLKTABLES_CONTINENTAL_SMALL: list(
        sorted(FOLKTABLES_SPLIT[FOLKTABLES_CONTINENTAL_SMALL].keys())
    ),
    FOLKTABLES_CONTINENTAL_LARGE: list(
        sorted(FOLKTABLES_SPLIT[FOLKTABLES_CONTINENTAL_LARGE].keys())
    ),
}


class PropertyEnumMeta(EnumMeta):
    """
    A custom metaclass for Enums.

    It adds a read-only properties to Enum (sort of like static properties)
    """

    @property
    def key(cls):
        def to_snake_case(camel_case_string: str) -> str:
            """
            Converts a CamelCase string to snake_case.

            Handles acronyms and consecutive capitals correctly.
            For example:
                'CamelCase' -> 'camel_case'
                'MyAPIResponse' -> 'my_api_response'
            """
            # Insert an underscore before any capital letter that is preceded by a lowercase letter.
            s1 = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", camel_case_string)

            # Insert an underscore between an acronym and the next word.
            # e.g., 'MyAPIResponse' -> 'My_API_Response'
            s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)

            return s2.lower()

        """Returns the name of the enum class as a title."""
        # 'cls' here refers to the class object itself (e.g., ModernStatus)
        return to_snake_case(cls.__name__)

    @property
    def members(cls):
        return cls.__members__

    @property
    def values(cls):
        return {item.value for item in cls}


class BaseEnhancedStrEnum(str, Enum, metaclass=PropertyEnumMeta):
    pass


class FedConfFairStage(BaseEnhancedStrEnum):
    INIT_STAGE = "init_stage"
    FED_CP = "fed_cp"
    POP_STAT = "pop_stat"
    CF_ITER = "cf_iter"

    @property
    def num(self):
        return {
            FedConfFairStage.INIT_STAGE: -1,
            FedConfFairStage.FED_CP: 0,
            FedConfFairStage.POP_STAT: 1,
            FedConfFairStage.CF_ITER: 2,
        }[self]


class QuantileMethod(BaseEnhancedStrEnum):
    TDIGEST = "tdigest"
    DDSKETCH = "ddsketch"
    MEAN = "mean"


class SpecialParameters(BaseEnhancedStrEnum):
    UPDATE_U_L = "update_u_l"
    UPDATE_POS = "update_positive_labels"


class FedCPFitConstants(str, Enum):
    SCORE_MODULE = "score_module"
    QUANT_SKETCH = "quant_sketch"
    Q_HAT = "q_hat"


class FedCFConstants(str, Enum):
    F_M = "fairness_metric"
    CLIENT_FORM = "client_formulation"


STATE_CONFIGS = "state_configs"
STATE_PARAMS = "state_params"
