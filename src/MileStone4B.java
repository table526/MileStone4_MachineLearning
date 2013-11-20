import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
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

public class MileStone4B {
 
    public static void main(String[] args) throws Exception{
        
    	String [] caseName = new String[24];
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
    	caseName[12] = "ionosphere";
    	caseName[13] = "labor";
    	caseName[14] = "lymph";
    	caseName[15] = "mushroom";
    	caseName[16] = "segment";
    	caseName[17] = "sonar";
    	caseName[18] = "soybean";
    	caseName[19] = "splice";
    	caseName[20] = "vehicle";
    	caseName[21] = "vote";
    	caseName[22] = "vowel";
    	caseName[23] = "zoo";
    	
    	String [] Options = new String[24];
    	
    	LMT lmt_cf = new LMT();
    	double [] error_0 = new double[24];
    	double [] error_1 = new double[24];
    	for(int i = 19; i < 20/*caseName.length*/; i++)
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
    		Options[i] = LMT_Option(train);
    		System.out.println(Options[i]);
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
       	PrintWriter writer = new PrintWriter("./predict/" + caseName + "0.predict");      
       	for (int i = 0; i < test.numInstances(); i++) {
       	   double pred = tree.classifyInstance(test.instance(i));
          	writer.println(Double.toString(pred));
       	}
     	writer.close();
     	
       	Evaluation eval = new Evaluation(train);
       	eval.evaluateModel(tree, test);

      	return eval.errorRate();
    } 
    
    public static String LMT_Option(Instances train) throws Exception
    {
        double W_val = 0.0; 
        int I_val = 0;
        CVParameterSelection ps = new CVParameterSelection();
        ps.setClassifier(new LMT());
        ps.setNumFolds(10);  // using 5-fold CV
        ps.addCVParameter("I -1 34 8");
        ps.addCVParameter("W 0.0 0.5 6");

        //ps.addCVParameter("I -1 14 4");

        // Get the value of W
        ps.buildClassifier(train);
        return Utils.joinOptions(ps.getBestClassifierOptions());
        /*  String [] opt_token = ps.getBestClassifierOptions();
        for(int i = 0; i < opt_token.length; i++)
        {
          if(opt_token[i].contains("W"))
          {
            W_val = Double.parseDouble(opt_token[i + 1]);
            break;
          }
        }
        
        ps = new CVParameterSelection();
        ps.setClassifier(new LMT());
        ps.setNumFolds(5);  // using 5-fold CV
        ps.addCVParameter("I -1 14 4");
        ps.addCVParameter("W " + Double.toString(W_val));
        ps.buildClassifier(train);
        for(int i = 0; i < opt_token.length; i++)
        {
          if(opt_token[i].contains("I"))
          {
            I_val = Integer.parseInt(opt_token[i + 1]);
            break;
          }
        }
        
        return "-W " + Double.toString(W_val) + " -I " + Integer.toString(I_val);*/
    }
    
    public static double LMT_model_revised(LMT tree, Instances train, Instances test, String Options, String caseName) throws Exception
    {
      	tree.setOptions(Options.split(" "));
        tree.buildClassifier(train);   // build classifier

        PrintWriter writer = new PrintWriter("./predict/" + caseName + "1.predict");      
        for (int i = 0; i < test.numInstances(); i++) {
        	double pred = tree.classifyInstance(test.instance(i));
        	writer.println(Double.toString(pred));
        }
        writer.close();
       	Evaluation eval = new Evaluation(train);
       	eval.evaluateModel(tree, test);
       	
      	return eval.errorRate();
    } 


    
}