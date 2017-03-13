/**
 * Created by jw on 11-3-17.
 */



public class mlp {

    public static void main(String[] args){
        double[][] mat = { { 1, 2, 3 }, { 4, 5, 6 }};
        Matrix A = new Matrix(mat);

        A.multiply(2);
        A.print();


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

        Matrix w_hidden = Matrix.random(n_input + 1, n_hidden);
        w_hidden.multiply(weight_spread - (weight_spread/2) + mean_weight);

        Matrix w_output = Matrix.random(n_hidden, n_output);
        w_output.multiply(weight_spread - (weight_spread/2) + mean_weight);

        double min_error = 0.01;
        boolean stop_criterium = false;
        int epoch = 0;

        while(!stop_criterium){

            Matrix noise = Matrix.random(examples.numRows(),examples.numCols());
            noise.multiply(noise_level);

            Matrix input_data = examples.add(noise);

            

            stop_criterium = true;

        }


    }
}


