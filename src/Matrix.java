import org.omg.SendingContext.RunTime;

import java.util.Random;

/**
 * Created by jw on 11-3-17.
 */
public class Matrix implements java.io.Serializable{
    private int m; //rows
    private int n; //columns
    public double[][] data;

    public Matrix(int m, int n){
        this.m = m;
        this.n = n;

        data = new double[m][n];
    }

    public Matrix(double[][] data) {
        m = data.length;
        n = data[0].length;
        this.data = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                this.data[i][j] = data[i][j];
    }

    public static Matrix random(int m, int n) {
        Random random = new Random();
        Matrix A = new Matrix(m, n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                A.data[i][j] = random.nextDouble()*2 - 1;
        return A;
    }

    public static Matrix ones(int m, int n) {
        Matrix A = new Matrix(m, n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                A.data[i][j] = 1.0;
        return A;
    }

    public static Matrix random01(int m, int n) {
        Random random = new Random();
        Matrix A = new Matrix(m, n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                A.data[i][j] = random.nextDouble();
        return A;
    }

    public static Matrix gaussian(int m, int n, double std) {
        Random random = new Random();
        Matrix A = new Matrix(m, n);
        double factor = Math.sqrt(Math.PI/8);
        double U = 0.0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                do {
                    U = random.nextDouble(); //uniform [0,1]
                } while (U==0.0||U==1.0);
                A.data[i][j] = std*factor*Math.log(U/(1-U));
            }
        return A;
    }

    public Matrix narrow01() {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++){
                if(this.data[i][j]<0) this.data[i][j] = 0;
                if(this.data[i][j]>1) this.data[i][j] = 1;
            }
        return this;
    }

    public Matrix transpose() {
        Matrix A = new Matrix(n, m);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                A.data[j][i] = this.data[i][j];
        return A;
    }

    // return C = A + B
    public Matrix add(Matrix B) {
        Matrix A = this;
        if (B.m != A.m || B.n != A.n) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }
        Matrix C = new Matrix(m, n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C.data[i][j] = A.data[i][j] + B.data[i][j];
        return C;
    }


    // return C = A - B
    public Matrix sub(Matrix B) {
        Matrix A = this;
        if (B.m != A.m || B.n != A.n) throw new RuntimeException("Illegal matrix dimensions.");
        Matrix C = new Matrix(m, n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C.data[i][j] = A.data[i][j] - B.data[i][j];
        return C;
    }

    // does A = B exactly?
    public boolean equals(Matrix B) {
        Matrix A = this;
        if (B.m != A.m || B.n != A.n) throw new RuntimeException("Illegal matrix dimensions.");
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                if (A.data[i][j] != B.data[i][j]) return false;
        return true;
    }

    // return C = A * B
    public Matrix multiply(Matrix B) {
        Matrix A = this;
        if (A.n != B.m){
            throw new RuntimeException("Illegal matrix dimensions.");
        }
        Matrix C = new Matrix(A.m, B.n);
        for (int i = 0; i < C.m; i++)
            for (int j = 0; j < C.n; j++)
                for (int k = 0; k < A.n; k++)
                    C.data[i][j] += (A.data[i][k] * B.data[k][j]);
        return C;
    }

    public void printSize(){
        System.out.print(this.m);
        System.out.print(" ");
        System.out.println(this.n);
    }

    public Matrix multiply(double a) {
        Matrix A = new Matrix(this.m,this.n);
        for (int i = 0; i < this.m; i++) {
            for (int j = 0; j < this.n; j++) {
                A.data[i][j] = this.data[i][j] * a;
            }
        }
        return A;
    }

    public Matrix addBias(double bias_value){
        Matrix A = new Matrix(this.m, this.n+1);
        for (int i = 0; i < this.m; i++) {
            for (int j = 0; j < this.n; j++) {
                A.data[i][j] = this.data[i][j];
            }
        }
        for (int k = 0; k < A.m; k++) {
            A.data[k][A.n-1] = bias_value;
        }
        return A;
    }

    public void print() {
        System.out.println("Start of matrix");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(data[i][j]);
                System.out.print(" ");
            }
            System.out.println();
        }
        System.out.println("End of matrix");
        System.out.println();
    }

    public Matrix getRow(int index) {
        Matrix A = new Matrix(1,this.n);
        for (int i = 0; i < this.n; i++){
            A.data[0][i] = this.data[index][i];
        }
        return A;
    }
    public Matrix getCol(int index) {
        Matrix A = new Matrix(this.m,1);
        for (int i = 0; i < this.m; i++){
            A.data[i][0] = this.data[i][index];
        }
        return A;
    }

    public int numRows(){
        return this.m;
    }

    public int numCols(){
        return this.n;
    }

    public Matrix sigmoid(){
        Matrix A = new Matrix(this.m,this.n);
        for (int j = 0; j < this.n; j++){
            A.data[0][j] = (1/( 1 + Math.pow(Math.E,(-1*this.data[0][j]))));;
        }
        return A;
    }

    public Matrix d_sigmoid(){
        Matrix A = this.sigmoid();
        Matrix B = new Matrix(this.m,this.n);
        for (int j = 0; j < this.n; j++){
            B.data[0][j] = A.data[0][j] * (1 - A.data[0][j]);
        }
        return B;
    }


    public Matrix multiplyElementwise(Matrix B){
        Matrix A = this;
        if (B.m != A.m || B.n != A.n) throw new RuntimeException("Illegal matrix dimensions.");
        Matrix C = new Matrix(A.m, A.n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C.data[i][j] = A.data[i][j] * B.data[i][j];
        return C;

    }

    public static Matrix concat(Matrix A, Matrix B){ //function for combining two matrices with the same amount of columns
        if (A.n != B.n) throw new RuntimeException("Amount of columns is not the same");
        Matrix C = new Matrix(A.m + B.m, A.n);
        for (int i = 0; i < A.m; i++){
            for (int j = 0; j < C.n; j++) {
                C.data[i][j] = A.data[i][j];
            }
        }
        for (int i = 0; i < B.m; i++){
            for (int j = 0; j < C.n; j++){
                C.data[i+A.m][j] = B.data[i][j];
            }
        }
        return C;
    }

    public void set(int m, int n, double value){
        this.data[m][n] = value;
    }

    public double get(int m, int n){
        return this.data[m][n];
    }

}
