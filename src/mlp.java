import java.util.ArrayList;

/**
 * Created by jw on 11-3-17.
 */



public class mlp implements java.io.Serializable {
    public double learn_rate = 0.0005;
    public double mean_weight = 0;
    public double weight_spread = 2;

    public int n_input;
    public int n_hidden = 200;
    public int n_output = 1;

    public double noise_level = 0.01;
    public double bias_value = -1.0;

    public enum activationFunction {
        LINEAR, SIGMOID
    }
    public activationFunction actFunc;

    Matrix w_hidden;
    Matrix w_output;


    public mlp(int n_input){
        //Randomly initialise weights
        w_hidden = Matrix.random(n_input + 1, n_hidden);
        w_output = Matrix.random(n_hidden, n_output);
        w_hidden = w_hidden.multiply(weight_spread - (weight_spread / 2) + mean_weight);
        w_output = w_output.multiply(weight_spread - (weight_spread / 2) + mean_weight);
        this.actFunc = activationFunction.SIGMOID;

        this.n_input = n_input;
    }
    public mlp(int n_input, activationFunction actFunc){
        //Randomly initialise weights
        w_hidden = Matrix.random(n_input + 1, n_hidden);
        w_output = Matrix.random(n_hidden, n_output);
        w_hidden = w_hidden.multiply(weight_spread - (weight_spread / 2) + mean_weight);
        w_output = w_output.multiply(weight_spread - (weight_spread / 2) + mean_weight);
        this.actFunc = actFunc;

        this.n_input = n_input;
    }

    public mlp(int n_input, double learn_rate){
        //Randomly initialise weights
        w_hidden = Matrix.random(n_input + 1, n_hidden);
        w_output = Matrix.random(n_hidden, n_output);
        w_hidden = w_hidden.multiply(weight_spread - (weight_spread / 2) + mean_weight);
        w_output = w_output.multiply(weight_spread - (weight_spread / 2) + mean_weight);
        this.actFunc = activationFunction.SIGMOID;

        this.n_input = n_input;
        this.learn_rate = learn_rate;
    }
    public mlp(int n_input, activationFunction actFunc, double learn_rate){
        //Randomly initialise weights
        w_hidden = Matrix.random(n_input + 1, n_hidden);
        w_output = Matrix.random(n_hidden, n_output);
        w_hidden = w_hidden.multiply(weight_spread - (weight_spread / 2) + mean_weight);
        w_output = w_output.multiply(weight_spread - (weight_spread / 2) + mean_weight);
        this.actFunc = actFunc;

        this.n_input = n_input;
        this.learn_rate = learn_rate;
    }



    public void train(Matrix examples, Matrix goal, double learning_factor) {

        //Matrix examples = new Matrix(new double[][] {{0,0},{1,0},{0,1},{1,1}});
        //Matrix goal = new Matrix(new double[][] {{0.01},{0.99},{0.99},{0.01}});

        Matrix noise = Matrix.random(examples.numRows(), examples.numCols());
        noise = noise.multiply(noise_level);

        Matrix input_data = examples.add(noise);

        input_data = input_data.addBias(bias_value);
        //input_data.print();

        for (int pattern = 0; pattern < input_data.numRows(); pattern++) {
            //System.out.println(pattern);

            Matrix hidden_activation = input_data.getRow(pattern);

            hidden_activation = hidden_activation.multiply(w_hidden);

            Matrix hidden_output;
            switch (actFunc){
                case SIGMOID:
                    hidden_output = hidden_activation.sigmoid();
                    break;
                case LINEAR:
                    hidden_output =  hidden_activation.sigmoid();
                    break;
                default:
                    hidden_output = hidden_activation.sigmoid();
                    break;
            }

            Matrix output_activation = hidden_output.multiply(w_output);

            //1x1 matrix
            Matrix outputM;
            switch (actFunc){
                case SIGMOID:
                    outputM = output_activation.sigmoid();
                    break;
                case LINEAR:
                    outputM = output_activation;
                    break;
                default:
                    outputM = output_activation.sigmoid();
                    break;
            }

            double output = outputM.data[0][0];

            double output_error = goal.data[pattern][0] - output;

            Matrix local_gradient_output;
            switch (actFunc){
                case LINEAR: //fill with zeroes
                    local_gradient_output = Matrix.ones(output_activation.numRows(), output_activation.numCols());
                    break;
                case SIGMOID:
                    local_gradient_output = output_activation.d_sigmoid();
                    break;
                default:
                    local_gradient_output = output_activation.d_sigmoid();
                    break;
            }

            local_gradient_output = local_gradient_output.multiply(output_error);

            Matrix hidden_error = w_output.multiply(local_gradient_output.data[0][0]);

            Matrix local_gradient_hidden;
            switch (actFunc){
                case LINEAR: //fill with zeroes
                    local_gradient_hidden = hidden_activation.d_sigmoid();
                    break;
                case SIGMOID:
                    local_gradient_hidden = hidden_activation.d_sigmoid();
                    break;
                default:
                    local_gradient_hidden = hidden_activation.d_sigmoid();
                    break;
            }


            local_gradient_hidden = local_gradient_hidden.multiplyElementwise(hidden_error.transpose());


            Matrix delta_output = hidden_output.multiply(local_gradient_output.data[0][0]);
            delta_output = delta_output.multiply(learn_rate*learning_factor);

            Matrix delta_hidden = input_data.getRow(pattern).transpose().multiply(local_gradient_hidden);
            delta_hidden = delta_hidden.multiply(learn_rate*learning_factor);

            w_hidden = w_hidden.add(delta_hidden);

            w_output = w_output.add(delta_output.transpose());
            //delta_output.transpose().print();

        }


    }

    public Matrix output(Matrix input){
        Matrix input_data = input.addBias(bias_value);

        Matrix hidden_activation = input_data;
        hidden_activation = hidden_activation.multiply(w_hidden);


        Matrix hidden_output;
        switch (actFunc){
            case SIGMOID:
                hidden_output = hidden_activation.sigmoid();
                break;
            case LINEAR:
                hidden_output =  hidden_activation.sigmoid();
                break;
            default:
                hidden_output = hidden_activation.sigmoid();
                break;
        }

        //hidden_output.print();

        Matrix output_activation = hidden_output.multiply(w_output);

        //1x1 matrix
        Matrix outputM;
        switch (actFunc){
            case SIGMOID:
                outputM = output_activation.sigmoid();
                break;
            case LINEAR:
                outputM = output_activation;
                break;
            default:
                outputM = output_activation.sigmoid();
                break;
        }
        //outputM.print();
        //double output = outputM.data[0][0];
        return outputM;
    }




}


