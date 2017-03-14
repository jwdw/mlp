/**
 * Created by jw on 11-3-17.
 */



public class mlp {


    public static void main(String[] args){
        double[][] mat = { { 1, 2, 3 }, { 4, 5, 6 }};
        Matrix A = new Matrix(mat);

        Matrix D = new Matrix(new double[][] {{1,2,3,4}});
        Matrix E = new Matrix(new double[][] {{1,2,3,4}});

        Matrix F = D.multiply(E.transpose());
        F.print();


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
        w_hidden.multiply(weight_spread - (weight_spread/2) + mean_weight);

        Matrix w_output = Matrix.random(n_hidden, n_output);
        w_output.multiply(weight_spread - (weight_spread/2) + mean_weight);

        double min_error = 0.01;
        boolean stop_criterium = false;
        int epoch = 0;

        while(!stop_criterium){

            //noise not correct yet
            Matrix noise = Matrix.random(examples.numRows(),examples.numCols());
            noise.multiply(noise_level);

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
                local_gradient_output.multiply(output_error);

                local_gradient_output.print();

                Matrix local_gradient_hidden = hidden_activation.d_sigmoid();


            }


            stop_criterium = true;

        }


    }
}


