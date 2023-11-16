import sciann as sn
from model_ode_solver import *

class NeuralNetworkModel:
    def __init__(self, network_architecture, activation):
        self.network_architecture = network_architecture
        self.activation = activation
        self.model = None
        self.learnable_parameters = None
    
    def create_model(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    # Train the model
    def train(self, model_input, data_constraints, epochs=500, batch_size=25,
              shuffle=True, learning_rate=0.001, reduce_lr_after=100,
              stop_loss_value=1e-8, verbose=0):
        if self.model is None:
            raise ValueError("The model has not been created yet. Call create_model first.")

        input_data = [model_input]
        target_data = self.get_target_data(data_constraints)

        history = self.model.train(
            x_true=input_data,
            y_true=target_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            learning_rate=learning_rate,
            reduce_lr_after=reduce_lr_after,
            stop_loss_value=stop_loss_value,
            verbose=verbose
        )
        return history

    def get_target_data(self, data_constraints):
        raise NotImplementedError("Subclasses should implement this method.")

# Child class: Saturated Growth NN Model(NN_model)
class SaturatedGrowthNNModel(NeuralNetworkModel):
    def create_model(self):
        t = sn.Variable("t", dtype='float64')
        u = sn.Functional("u", t, self.network_architecture, self.activation)
        C = sn.Parameter(0.5, inputs=t, name="C")
        u_t = sn.diff(u, t)
        d1 = sn.Data(u)
        c1 = sn.Tie(u_t, u*(C-u))
        self.model = sn.SciModel(t, [d1, c1])
        self.learnable_parameters = [C]

    def get_target_data(self, data_constraints):
        # For the saturated growth model:'data_constraints' contains the data for 'u'.
        data_d1 = data_constraints
        data_c1 = 'zeros'  # The ODE constraint for the growth model is set to zero.
        
        # list ot constraints as Target
        target_data = [data_d1, data_c1]
        return target_data

class CompetitionNNModel(NeuralNetworkModel):
    def __init__(self, network_architecture, activation, higher_order=False):
        super().__init__(network_architecture, activation)
        self.higher_order = higher_order

    def create_model(self):
        t = sn.Variable("t", dtype='float64')
        u = sn.Functional("u", t, self.network_architecture, self.activation)
        v = sn.Functional("v", t, self.network_architecture, self.activation)
        u_t = sn.diff(u, t)
        v_t = sn.diff(v, t)
        d1 = sn.Data(u)
        d2 = sn.Data(v)

        # Initialize the target parameters.
        parameters = {
            "r": sn.Parameter(0.5, inputs=t, name="r"),
            "a1": sn.Parameter(0.5,inputs=t, name="a1"),
            "a2": sn.Parameter(0.5, inputs=t, name="a2"),
            "b1": sn.Parameter(0.5, inputs=t, name="b1"),
            "b2": sn.Parameter(0.5, inputs=t, name="b2")
        }

        # Define model constraints without higher-order interactions.
        c1 = sn.Tie(u_t, u*(1-parameters["a1"]*u-parameters["a2"]*v))
        c2 = sn.Tie(v_t, parameters["r"]*v*(1-parameters["b1"]*u-parameters["b2"]*v))

        # If higher-order==true
        if self.higher_order:
            # Additional parameters for higher-order interactions.
            for coef in ["e0", "e1", "e2", "e3", "e4", "f0", "f1", "f2", "f3", "f4"]:
                parameters[coef] = sn.Parameter(0.2, inputs=t, name=coef)

            # Modify model constraints to include higher-order terms.
            c1 = sn.Tie(u_t, u*(1-parameters["a1"]*u-parameters["a2"]*v) +
                        parameters["e0"] + parameters["e1"]*u + parameters["e2"]*v +
                        parameters["e3"]*u*u*v + parameters["e4"]*u*v*v)
            c2 = sn.Tie(v_t, parameters["r"]*v*(1-parameters["b1"]*u-parameters["b2"]*v) +
                        parameters["f0"] + parameters["f1"]*u + parameters["f2"]*v +
                        parameters["f3"]*u*u*v + parameters["f4"]*u*v*v)

        # Compile the model.
        self.model = sn.SciModel(t, [d1, d2, c1, c2])
        self.learnable_parameters = list(parameters.values())

    def get_target_data(self, data_constraints):
        # For the saturated growth model:'data_constraints' contains the data for 'u' & 'v'.
        data_d1, data_d2 = data_constraints[:,0], data_constraints[:,1]
        # The ODE constraint for the growth model is set to zero.
        data_c1 = 'zeros'  
        data_c2 = 'zeros'
        
        # list ot constraints as Target
        target_data = [data_d1, data_d2, data_c1, data_d2]
        return target_data


# if __name__=="__main__":
    
#     comp_model_params = [0.5, 0.3, 0.6, 0.7, 0.3, 0,0, 0,0 ,0,0, 0,0, 0,0]
#     comp_model = CompetitionModel(comp_model_params)

#     # Training data
#     model_input, data_constraints = create_train_data(comp_model, tend=24, 
#                                                       initial_conditions=[2,1], numpoints=50, 
#                                                       show_figure=True, time_limit=[5,15], 
#                                                       noise_level=0.05)

#     # Set SciANN model
#     network_architecture = 3*[10]
#     activation = 'tanh'
#     higher_order = True

#     comp_nn_model = CompetitionNNModel(network_architecture=network_architecture,
#                                        activation=activation, 
#                                        higher_order=higher_order)
#     comp_nn_model.create_model()

#     sci_model = comp_nn_model.model
#     learnable_parameters = comp_nn_model.learnable_parameters

#     history = comp_nn_model.train(
#                                     model_input=model_input, 
#                                     data_constraints=data_constraints,
#                                     epochs=2000,
#                                     batch_size=25,
#                                     verbose=1  
#                                 )
    
#     learned_parameter_values = {
#                                 param.name: param.eval([model_input])
#                                 for param in comp_nn_model.learnable_parameters
#                                 }
#     # for name, value in learned_parameter_values.items():
#     #     print(f"{name}:{value}")
#     # True parameter names for reference (adjust as needed to match your model's parameters)
#     true_parameter_names = ["r", "a1", "a2", "b1", "b2", "e0", "f0", "e1", "f1", "e2", "f2", "e3", "f3", "e4", "f4"]

#     # Print the true parameters along with the learned parameter values
#     for name, true_value in zip(true_parameter_names, comp_model_params):
#         learned_value = learned_parameter_values.get(name)
#         print(f"{name}: True Value = {true_value}, Learned Value = {learned_value}")


