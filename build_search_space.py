#%%
import typing as ty
import itertools as itools
#%%

class SearchSpaceBuilder:

    def __init__(self, constant_keys:list=None):
        if not constant_keys:
            constant_keys = []
        self._constant_keys = constant_keys

    def turn_into_list(self, key:str, val:object) -> ty.Iterable:
        if SearchSpaceBuilder.check_for_list(val) and key not in self._constant_keys:
            return list(val)
        else:
            return [val]

    def build(self, param_dict:dict) -> ty.Tuple:
        keys = param_dict.keys()
        vals = list(map(lambda pair: self.turn_into_list(*pair), param_dict.items()))
        product_space = itools.product(*vals)
        for space_values in product_space:
            yield dict(zip(keys, space_values))

    @staticmethod
    def check_for_list(x:object) -> bool:
        iter_bool = isinstance(x, ty.Iterable)
        str_bool = isinstance(x, str)
        
        full_bool = iter_bool and not (str_bool)
        return full_bool
    
