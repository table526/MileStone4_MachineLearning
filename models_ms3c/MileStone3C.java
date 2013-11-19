import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.LMT;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class MileStone3C {
 
    public static void main(String[] args) throws Exception{
        
    	String [] caseName = new String[12];
    	caseName[0] = "anneal";
    	caseName[1] = "audiology";
    	caseName[2] = "autos";
    	caseName[3] = "balance-scale";
    	caseName[4] = "breast-cancer";
    	caseName[5] = "colic";
    	caseName[6] = "credit-a";
    	caseName[7] = "diabetes";
    	caseName[8] = "glass";
    	caseName[9] = "heart-c";
    	caseName[10] = "hepatitis";
    	caseName[11] = "hypothyroid2";
    	
    	String [] Options = new String[12];
    	Options[0] = "-I -1 -M 15 -W 0.0";
    	Options[1] = "-I 15 -M 15 -W 0.1";
    	Options[2] = "-I 30 -M 15 -W 0.0";
    	Options[3] = "-I 30 -M 15 -W 0.2";
    	Options[4] = "-I -1 -M 15 -W 0.2";
    	Options[5] = "-I 15 -M 15 -W 0.0";
    	Options[6] = "-I -1 -M 15 -W 0.2";
    	Options[7] = "-I 15 -M 15 -W 0.2";
    	Options[8] = "-I -1 -M 15 -W 0.0";
    	Options[9] = "-I -1 -M 15 -W 0.0";
    	Options[10] = "-I -1 -M 15 -W 0.2";
    	Options[11] = "-I 4 -M 15 -W 0.0";
    	
    	LMT lmt_cf = new LMT();
    	double [] error_0 = new double[12];
    	double [] error_1 = new double[12];
    	for(int i = 0; i < caseName.length; i++)
    	{
    		DataSource train_source = new DataSource(caseName[i] + "_train.arff");
    		Instances train = train_source.getDataSet();
    		// setting class attribute if the data format does not provide this information
    		// For example, the XRFF format saves the class attribute information as well
    		if (train.classIndex() == -1)
    			train.setClassIndex(train.numAttributes() - 1);

   	 
    		DataSource test_source = new DataSource(caseName[i] + "_test.arff");
    		Instances test = test_source.getDataSet();
    		// setting class attribute if the data format does not provide this information
    		// For example, the XRFF format saves the class attribute information as well
    		if (test.classIndex() == -1)
    			test.setClassIndex(train.numAttributes() - 1);
   	    	
    		NaiveBayes nb_cf = new NaiveBayes();
   	 
    		double nb_error = nb_model(nb_cf, train, test);
    		error_0[i] = LMT_model(lmt_cf, train, test, caseName[i]) / nb_error;
    		error_1[i] = LMT_model_revised(lmt_cf, train, test, Options[i], caseName[i]) / nb_error;
    	}  	
    		
    	for(int i = 0; i < caseName.length; i++)
    	{
    		System.out.println(Double.toString(error_0[i]) + "\t" + Double.toString(error_1[i]));
    	}
    }
    	
    
    
    public static double nb_model(NaiveBayes nb_cf, Instances train, Instances test) throws Exception
    {
       	nb_cf.buildClassifier(train);   // build classifier
       	
       	Evaluation eval = new Evaluation(train);
       	eval.evaluateModel(nb_cf, test);
      	return eval.errorRate();
    }
    
    public static double LMT_model(LMT tree, Instances train, Instances test, String caseName) throws Exception
    {
    	tree.buildClassifier(train);   // build classifier
       	
       	Evaluation eval = new Evaluation(train);
       	eval.evaluateModel(tree, test);
		ObjectOutputStream oos = new ObjectOutputStream(
				new FileOutputStream("./models/" + caseName + "0.model"));
		oos.writeObject(tree);
		oos.flush();
		oos.close();
      	return eval.errorRate();
    } 
    
    public static double LMT_model_revised(LMT tree, Instances train, Instances test, String Options, String caseName) throws Exception
    {
      	tree.setOptions(Options.split(" "));
    	tree.buildClassifier(train);   // build classifier
       	
       	Evaluation eval = new Evaluation(train);
       	eval.evaluateModel(tree, test);
       	ObjectOutputStream oos = new ObjectOutputStream(
				new FileOutputStream("./models/" + caseName + "1.model"));
		oos.writeObject(tree);
		oos.flush();
		oos.close();
      	return eval.errorRate();
    } 


    
}