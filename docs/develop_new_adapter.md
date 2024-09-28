## How to develop a new adapter

Step1. Start by creating the corresponding adapter configuration data item in `config/adapter.py` This will serve as the foundational step in defining your adapter.

Step2. Create your module definition in the `model/modules` folder, like `lora.py` file.

Step3. Define the forward propagation of the adapter in `model/modules/linear.py`.

Step4. Create functions for loading and storing the weights of the adapter in `executor/context`, referring to `vera.py`.