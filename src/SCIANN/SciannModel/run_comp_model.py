import sciann as sn
from model_sciann import NeuralNetworkModel, SaturatedGrowthNNModel, CompetitionNNModel
from model_ode_solver import *

def run_competition_model(survival=True):
    # Set parameters for the competition model
    if survival:
        comp_model_params = [0.5, 0.3, 0.6, 0.7, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        comp_model_params = [0.5, 0.7, 0.3, 0.3, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    comp_model = CompetitionModel(comp_model_params)

    time_limit=[0,30]
    # Training data
    model_input, data_constraints = create_train_data(comp_model, tend=24, 
                                                      initial_conditions=[2,1], numpoints=100, 
                                                      show_figure=False, time_limit=time_limit, 
                                                      noise_level=0.00)

    # Set SciANN model
    network_architecture = 3*[5]
    activation = 'tanh'
    higher_order = False

    comp_nn_model = CompetitionNNModel(network_architecture=network_architecture,
                                       activation=activation, 
                                       higher_order=higher_order)
    comp_nn_model.create_model()

    # Train the model
    history = comp_nn_model.train(
                                    model_input=model_input, 
                                    data_constraints=data_constraints,
                                    epochs=200,
                                    batch_size=25,
                                    verbose=1
                                )
    
    # Evaluate learned parameters
    learned_parameter_values = {
                                param.name: param.eval([model_input])
                                for param in comp_nn_model.learnable_parameters
                                }

    # Print learned parameter values
    true_parameter_names = ["r", "a1", "a2", "b1", "b2", "e0", "f0", "e1", "f1", "e2", "f2", "e3", "f3", "e4", "f4"]

    # Print the true parameters along with the learned parameter values
    for name, true_value in zip(true_parameter_names, comp_model_params):
        learned_value = learned_parameter_values.get(name)
        print(f"{name}: True Value = {true_value}, Learned Value = {learned_value}")

    # True output
    time, sol = create_train_data(comp_model, tend=24,
                                    initial_conditions=[2,1],
                                    show_figure=False)
    u_ture, v_ture = sol[:,0], sol[:,1]
    # predict
    tspan = np.arange(0,24, 0.1)
    prediction = comp_nn_model.model.predict(tspan)
    plt.plot(tspan, prediction[0], label='u_predict')
    plt.plot(tspan, prediction[1], label='v_predict')
    plt.plot(time, u_ture, label='u_ture')
    plt.plot(time, v_ture, label='v_true')
    plt.title(f'Training Time: {time_limit}') 
    plt.legend()
    plt.show()

# Call the test function
if __name__ == "__main__":
    run_competition_model(survival=False)
