import sciann as sn
from model_sciann import NeuralNetworkModel, SaturatedGrowthNNModel, CompetitionNNModel
from model_ode_solver import *

def run_sg_model():
    sg_model_params = [1]
    sg_model = SaturatedGrowthModel(sg_model_params)
    time_limit = [5,15]

    # Training Data
    model_input, data_constraints = create_train_data(sg_model, tend=24,
                                                      initial_conditions=0.01, numpoints=100,
                                                      show_figure=False, time_limit=time_limit,
                                                      noise_level=0.005)
    # SciANN model
    network_architecture = 3*[5]
    activation = 'tanh'

    sg_nn_model = SaturatedGrowthNNModel(network_architecture=network_architecture,
                                         activation=activation)
    sg_nn_model.create_model()

    # Train the model
    history = sg_nn_model.train(model_input=model_input,
                                data_constraints=data_constraints,
                                epochs=1000,
                                batch_size=25,
                                verbose=1)
    # Evaluate learned parameters
    learned_parameter_values = {
                                param.name: param.eval([model_input])
                                for param in sg_nn_model.learnable_parameters
                                }

    true_parameter_names = ["C"]
    # Print the true parameters along with the learned parameter values
    for name, true_value in zip(true_parameter_names, sg_model_params):
        learned_value = learned_parameter_values.get(name)
        print(f"{name}: True Value = {true_value}, Learned Value = {learned_value}")

    # True output
    time, u_ture = create_train_data(sg_model, tend=24,
                                    initial_conditions=0.01,
                                    show_figure=False)
    # predict
    tspan = np.arange(0,24, 0.1)
    prediction = sg_nn_model.model.predict(tspan)
    plt.plot(tspan, prediction[0], label='Predicted')
    plt.plot(time, u_ture, label='True Value')
    plt.title(f'Training Time: {time_limit}') 
    plt.legend()
    #plt.plot(tspan, prediction[1])
    plt.show()

if __name__=="__main__":
    run_sg_model()