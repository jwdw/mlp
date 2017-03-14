import java.util.ArrayList;

/**
 * Created by jw on 11-3-17.
 */



public class mlp {


    public static void main(String[] args){
        double[][] mat = { { 1, 2, 3 }, { 4, 5, 6 }};
        Matrix A = new Matrix(mat);

        Matrix D = new Matrix(new double[][] {{1,2,3,4}});
        Matrix E = new Matrix(new double[][] {{1,2,3,4}});

        Matrix F = D.multiplyElementwise(E);
        //F.print();


        Matrix examples = new Matrix(new double[][] {{0,0},{1,0},{0,1},{1,1}});

        Matrix goal = new Matrix(new double[][] {{0.01},{0.99},{0.99},{0.01}});

        double learn_rate = 0.2;
        int max_epoch = 5000;

        double mean_weight = 0;
        double weight_spread = 5;

        int n_input = examples.numCols();
        int n_hidden = 20;
        int n_output = goal.numCols();

        double noise_level = 0.01;
        double bias_value = -1.0;

        Matrix w_hidden = Matrix.random(n_input + 1, n_hidden);
        w_hidden = w_hidden.multiply(weight_spread - (weight_spread/2) + mean_weight);

        Matrix w_output = Matrix.random(n_hidden, n_output);
        w_output = w_output.multiply(weight_spread - (weight_spread/2) + mean_weight);

        double min_error = 0.01;
        boolean stop_criterium = false;
        int epoch = 0;

        ArrayList h_error = new ArrayList();

        while(!stop_criterium){
            epoch += 1;

            //noise not correct yet
            Matrix noise = Matrix.random(examples.numRows(),examples.numCols());
            noise = noise.multiply(noise_level);

            Matrix input_data = examples.add(noise);

            input_data = input_data.addBias(bias_value);
            //input_data.print();

            double epoch_error = 0;
            double epoch_delta_hidden = 0;
            double epoch_delta_output = 0;

            for (int pattern = 0; pattern < input_data.numRows(); pattern++){

                Matrix hidden_activation = input_data.getRow(pattern);
                hidden_activation = hidden_activation.multiply(w_hidden);

                Matrix hidden_output = hidden_activation.sigmoid();

                Matrix output_activation = hidden_output.multiply(w_output);

                //1x1 matrix
                Matrix outputM = output_activation.sigmoid();
                double output = outputM.data[0][0];

                double output_error = goal.data[pattern][0] - output;

                Matrix local_gradient_output = output_activation.d_sigmoid();
                local_gradient_output = local_gradient_output.multiply(output_error);

                Matrix hidden_error = w_output.multiply(local_gradient_output.data[0][0]);

                Matrix local_gradient_hidden = hidden_activation.d_sigmoid();
                local_gradient_hidden = local_gradient_hidden.multiplyElementwise(hidden_error.transpose());


                Matrix delta_output = hidden_output.multiply(local_gradient_output.data[0][0]);
                delta_output = delta_output.multiply(learn_rate);


                Matrix delta_hidden = input_data.getRow(pattern).transpose().multiply(local_gradient_hidden);
                delta_hidden = delta_hidden.multiply(learn_rate);

                w_hidden = w_hidden.add(delta_hidden);

                w_output = w_output.add(delta_output.transpose());

                epoch_error += output_error * output_error;

            }

            h_error.add(epoch_error / input_data.numRows());

            if (epoch >= max_epoch || epoch_error < min_error){
                stop_criterium = true;
            }

        }
        System.out.println(epoch);
        System.out.println(h_error);


    }
}


