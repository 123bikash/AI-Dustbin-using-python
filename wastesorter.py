from wastemodule import WasteSorter

class_dic = {
    0: None,
    1: 0,
    2: 0,
    3: 1,
    4: 1
}

waste_sorter = WasteSorter(
    model_path="Model/keras_model.h5",
    label_path="Model/labels.txt",
    waste_folder="Waste",
    bins_folder="Bins",
    arrow_path="arrow.png",
    background_path="background.png",
    class_dic=class_dic 
)


waste_sorter.run()
