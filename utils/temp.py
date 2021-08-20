import pdb
import matplotlib.pyplot as plt
from matplotlib import cm

my_dict = {"Cup":{"Background":1491,"Toaster":15,"Cup":114,"Bowl":72,"SoapBottle":85,"GarbageCan":353,"Tomato":3,"Mug":164,"Bread":13,"Pan":3,"Pot":34,"PepperShaker":7,"SaltShaker":16,"Apple":1,"Lettuce":3,"CoffeeMachine":5,"Potato":7,"Faucet":1,"Sink":3},
"Pot":{"Background":1024,"Pot":233,"Sink":98,"Pan":161,"Toaster":304,"Apple":4,"StoveBurner":78,"GarbageCan":99,"Bowl":283,"Plate":2,"Cup":4,"CoffeeMachine":18,"Bread":14,"ButterKnife":2,"Knife":4,"Lettuce":4},
"Mug":{"Mug":173,"GarbageCan":83,"Background":1699,"Tomato":119,"Potato":10,"Bread":21,"Lettuce":28,"Cup":249,"Bowl":32,"SoapBottle":5,"Pot":37,"Pan":3,"Apple":26,"SaltShaker":2,"CoffeeMachine":9,"StoveBurner":1,"Toaster":2,"Sink":2,"DishSponge":5},
"Egg":{"Apple":171,"Background":2061,"Tomato":293,"Cup":100,"Spatula":20,"DishSponge":40,"Egg":158,"Mug":41,"Bowl":6,"Potato":225,"Faucet":3,"StoveBurner":61,"GarbageCan":2,"Pot":1,"Sink":1,"Spoon":5},
"Spatula":{"Background":1860,"Apple":5,"DishSponge":237,"Pan":17,"ButterKnife":67,"Potato":17,"Knife":400,"Sink":37,"Bowl":4,"Spoon":163,"Pot":9,"StoveBurner":8,"CoffeeMachine":10,"GarbageCan":6,"Toaster":2,"Tomato":2,"Spatula":93,"Egg":141,"Fork":108,"SaltShaker":1,"Cup":78},
"Lettuce":{"Tomato":144,"Background":952,"Toaster":14,"Potato":1,"Bread":370,"StoveBurner":182,"Pot":7,"SaltShaker":2,"Mug":5,"Pan":62,"Cup":5,"Sink":6,"Lettuce":276,"Bowl":126,"Apple":25,"Knife":1,"CoffeeMachine":17,"Plate":36,"Egg":6,"GarbageCan":1,},
"Tomato":{"Bread":11,"Background":1490,"GarbageCan":69,"Bowl":139,"Cup":16,"Tomato":414,"Lettuce":9,"Apple":510,"Potato":199,"Toaster":1,"Mug":104,"SoapBottle":1,"CoffeeMachine":1,"Pot":1,"Egg":25,"Sink":2,"Plate":20,"Pan":1},
"Pan":{"Background":687,"Lettuce":26,"Bowl":472,"StoveBurner":516,"GarbageCan":58,"Cup":1,"Pot":92,"Pan":249,"Sink":15,"Plate":873,"CoffeeMachine":106,"Bread":38,"DishSponge":22,"Knife":11,"Toaster":15,"Tomato":2,"Apple":1,"Mug":1,"Spatula":2,"Potato":17,"Faucet":1},
"Bread":{"Background":1504,"Apple":1,"Pot":53,"GarbageCan":167,"Plate":14,"Lettuce":440,"Bowl":318,"Toaster":171,"Bread":426,"Tomato":24,"Pan":73,"Sink":63,"CoffeeMachine":2,"Mug":3,"Cup":3,"StoveBurner":38,"PepperShaker":1},
"Apple":{"Tomato":946,"Background":1202,"Apple":129,"Potato":136,"Lettuce":33,"Bread":35,"Pot":3,"CoffeeMachine":2,"Bowl":130,"GarbageCan":11,"Mug":122,"Sink":5,"Cup":18,"Spatula":4,"PepperShaker":2,"Egg":4}}


all_keys = set(my_dict.keys())
for v in my_dict.values():
    all_keys.update(set(v.keys()))
all_keys = list(all_keys)
all_keys = sorted(all_keys)

all_objects = list(my_dict.keys())
all_objects = sorted(all_objects)

import numpy as np
my_matrix = np.zeros((len(all_objects), len(all_keys)))
for k, v in my_dict.items():
    for l, number in v.items():
        if l == 'Background':
            continue
        my_matrix[all_objects.index(k), all_keys.index(l)] = number

original_matrix = my_matrix.copy()
import matplotlib.pyplot as plt
my_matrix = my_matrix /np.repeat((my_matrix.sum(axis=-1) + 1e-10).reshape(-1, 1), len(all_keys), axis=-1)
# plt.imshow(my_matrix)
fig, ax = plt.subplots()
ax.set_yticks([i for i in range(len(all_objects))])
ax.set_yticklabels(all_objects)
ax.set_xticks([i for i in range(len(all_keys))])
ax.set_xticklabels(all_keys)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
color_map = 'gist_earth'
color_map = 'gray'
im=ax.imshow(my_matrix, cmap=color_map)
# plt.colorbar(fig.colorbar(cm.ScalarMappable(norm=my_matrix.norm(), cmap=cmap), ax=ax))
# fig.colorbar(im, cax=ax, orientation='horizontal')
# plt.colorbar()
plt.show()
# pdb.set_trace()