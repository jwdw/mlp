/**
 * Created by s2616602 on 4/13/17.
 */
public class main {
    public static void main(String[] args) {
        /*
        mlplinear test = new mlplinear(1, mlplinear.activationFunction.SIGMOID);

        Matrix examples = new Matrix(new double[][] {{-1},{0},{0},{1}});
        Matrix goal = new Matrix(new double[][] {{0},{0.5},{0.5},{1}});
        Matrix output;
        int time = 0;

        while (true) {
            time++;
            test.train(examples,goal);
            if (time%10000 == 0){
                for (int pattern = 0; pattern < examples.numRows(); pattern++) {
                    output = test.output(examples.getRow(pattern));
                    System.out.println(output.get(0,0));
                }
                System.out.println();
                //Thread.sleep(1000);
            }
        }
        */


    mlp test = new mlp(1, mlp.activationFunction.LINEAR, 0.001);

    Matrix examples = new Matrix(new double[][] {{40},{-50},{10}});
    Matrix goal = new Matrix(new double[][] {{0.01},{0.5},{0.99}});

    double testnumber1;
    double testnumber2;
    double testoutput;

    int epochs = 100000;

    for (int i = 0; i < epochs; i++){
        if (i%(epochs/100) == 0)
            System.out.println(i / (epochs/100) + "%");

        testnumber1 = (Math.random() * 200) - 100;
        //testnumber2 = (Math.random() * 20) - 10;

        testoutput = testnumber1;
        //System.out.println("Testnumber: " + testnumber1 + " Testoutput: " + testoutput);

        test.train(new Matrix(new double[][] {{testnumber1}}), new Matrix(new double[][] {{testoutput}}),1.0);
    }

    //test.output(examples.getRow(3)).print();
    for (int j = -20; j < 20; j++){
        System.out.println(j);
        test.output(new Matrix(new double[][]{{j}})).print();
    }

    //test.output(new Matrix(new double[][] {{-1}})).print();

    }
}
