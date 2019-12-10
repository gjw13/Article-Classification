// Classification HW4
// Greg Wills
// Big Data Analytics

import scala.util.matching.Regex
import scala.xml.XML
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.optimization.L1Updater

object Classification {
    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("Wikipedia Article Classification").setMaster("local")
        val sc = new SparkContext(sparkConf)

        val x = 0
        var total_error_rate = 0.0
        val num_loops = 10
        var total_pos_instances: Long = 0
        var total_neg_instances: Long = 0
        var lowest_error_rate = 100.0
        var best_model_loop = 0

        val label_regex = """[0-1]""".r // using findFirstIn
	    val title_regex_training = """[^\(\t]+\w+""".r
        val title_regex_testing = """[^\(\t]+\w+""".r
        val article_regex = """[^\t]*$""".r

        // val rdd1 = sc.textFile("train0/part-0000[1-9]*")
        // val rdd2 = sc.textFile("train0/part-000[1][0-9]*")
        // val rdd3 = sc.textFile("train0/part-000[2][0-9]*")
        // val rdd4 = sc.textFile("train0/part-000[3][0-9]*")
        // val rdd5 = sc.textFile("train0/part-000[4][0-9]*")
        // val rdd6 = sc.textFile("train0/part-000[5][0-9]*")
        // val rdd7 = sc.textFile("train0/part-000[6][0-9]*")
        // val rdd8 = sc.textFile("train0/part-000[7][0-9]*")
        // val rdd9 = sc.textFile("train0/part-000[8][0-9]*")
        val rdd10 = sc.textFile("train0/part-000[0-2][0-9]*")

        // val rdd1 = sc.textFile("train0/part-00001")
        // val rdd2 = sc.textFile("train0/part-00002")
        // val rdd3 = sc.textFile("train0/part-00003")
        // val rdd4 = sc.textFile("train0/part-00004")
        // val rdd5 = sc.textFile("train0/part-00005")
        // val rdd6 = sc.textFile("train0/part-00006")
        // val rdd7 = sc.textFile("train0/part-00007")
        // val rdd8 = sc.textFile("train0/part-00008")
        // val rdd9 = sc.textFile("train0/part-00009")
        // val rdd10 = sc.textFile("train0/part-0001[0-9]*")
        //val cross_val_list = List(rdd1,rdd2,rdd3,rdd4,rdd5,rdd6,rdd7,rdd8,rdd9,rdd10)

        val lrLearner1 = new LogisticRegressionWithSGD()
        //lrLearner1.optimizer.setNumIterations(100).setRegParam(0.1).setUpdater(new L1Updater)
        //val learner_list = List(lrLearner1,lrLearner2,lrLearner3,lrLearner4,lrLearner5,lrLearner6,lrLearner7,lrLearner8,lrLearner9,lrLearner10)

        //var model_list = List()

        val testRdd = sc.textFile("testArticles/part-0[0-9]*")

        // Create a HashingTF instance to map email text to vectors of 10,000 features.
		val tf = new HashingTF(numFeatures = 3000)

		val myRDD = rdd10

        val label_article = myRDD.map {
        	case (inString) =>
        		val label = label_regex.findFirstIn(inString).mkString("")
        		val article = article_regex.findFirstIn(inString).mkString("")
        	(label, article)
        }

        //label_article.collect().foreach(println)

        val pos_test = label_article.filter(label => label._1.toInt == 1)
        val neg_test = label_article.filter(label => label._1.toInt == 0)

        // print positive labels count
        val pos_count = pos_test.count
        //total_pos_instances = total_pos_instances + pos_count

        // print positive labels count
        val neg_count = neg_test.count
        //total_neg_instances = total_neg_instances + neg_count

        /*************************
        / MACHINE LEARNING PART  *
        /************************/*/

        
		// Each email is split into words, and each word is mapped to one feature. 
		// val spamFeatures = spam.map(email => tf.transform(email.split(" ")))
		val normalFeatures = neg_test.map(f => tf.transform(f._2.toLowerCase.split(" ")))
		//normalFeatures.collect().foreach(println)
		val sportsFeatures = pos_test.map(f =>tf.transform(f._2.toLowerCase.split(" ")))
		//sportsFeatures.collect().foreach(println)
		// Create LabeledPoint datasets for positive (sports articles) and negative (normal) examples.
		val positiveExamples = sportsFeatures.map(features => LabeledPoint(1, features)) 
		val negativeExamples = normalFeatures.map(features => LabeledPoint(0, features)) 
		val trainingData = positiveExamples.union(negativeExamples)
		trainingData.cache() // Cache since Logistic Regression is an iterative algorithm.

        // Manual 10 fold cross validation
   //      for(x <- 1 to num_loops) {
   //      	//Split for 10 fold cross validation
			// val Array(split1,split2,split3,split4,split5,split6,split7,split8,split9,split10) = trainingData.randomSplit(Array(0.1, 0.1,0.1, 0.1,0.1, 0.1,0.1, 0.1,0.1, 0.1),seed=12)
			// //val Array(training,testing) = trainingData.randomSplit(Array(0.9,0.1))

			// if(x==1){
			// 	val training = sc.union(split2,split3,split4,split5,split6,split7,split8,split9,split10)
			// 	val testing = split1
			// 	training.cache()
			// 	testing.cache()
			// 	// Run the actual learning algorithm on the training data.
		 //    	val model = lrLearner1.run(training)

		 //    	// Evaluate model on training examples and compute training error
			// 	val valuesAndPreds = testing.map { point =>
		 //  			val prediction = model.predict(point.features)
		 //  			// val test = abs(point.label-prediction)
		 //  			(point.label, prediction)
			// 	}

			// 	val test = valuesAndPreds.map{ case(v,p) => math.abs(v-p)}
			// 	val incorrect_prediction = test.filter(label => label.toInt == 1)

			// 	// get error rate of model
			// 	val incorrect_count = incorrect_prediction.count.toDouble
			// 	val total_count = test.count.toDouble
			// 	val error_rate = (incorrect_count / total_count)*100

			// 	total_error_rate = total_error_rate + error_rate
			// 	total_pos_instances = total_pos_instances + pos_count
			// 	total_neg_instances = total_neg_instances + neg_count

			// 	println(s"Training Error Rate: $error_rate%")

			// 	if(error_rate<lowest_error_rate){
			// 		lowest_error_rate = error_rate
			// 		best_model_loop = x
			// 	}
			// }
			// else if(x==2){
			// 	val training = sc.union(split1,split3,split4,split5,split6,split7,split8,split9,split10)
			// 	val testing = split2
			// 	training.cache()
			// 	testing.cache()
			// 		// Run the actual learning algorithm on the training data.
		 //    	val model = lrLearner1.run(training)

		 //    	// Evaluate model on training examples and compute training error
			// 	val valuesAndPreds = testing.map { point =>
		 //  			val prediction = model.predict(point.features)
		 //  			// val test = abs(point.label-prediction)
		 //  			(point.label, prediction)
			// 	}

			// 	val test = valuesAndPreds.map{ case(v,p) => math.abs(v-p)}
			// 	val incorrect_prediction = test.filter(label => label.toInt == 1)

			// 	// get error rate of model
			// 	val incorrect_count = incorrect_prediction.count.toDouble
			// 	val total_count = test.count.toDouble
			// 	val error_rate = (incorrect_count / total_count)*100

			// 	total_error_rate = total_error_rate + error_rate
			// 	total_pos_instances = total_pos_instances + pos_count
			// 	total_neg_instances = total_neg_instances + neg_count

			// 	println(s"Training Error Rate: $error_rate%")

			// 	if(error_rate<lowest_error_rate){
			// 		lowest_error_rate = error_rate
			// 		best_model_loop = x
			// 	}
			// }
			// else if(x==3){
			// 	val training = sc.union(split1,split2,split4,split5,split6,split7,split8,split9,split10)
			// 	val testing = split3
			// 	training.cache()
			// 	testing.cache()
			// 	val model = lrLearner1.run(training)

		 //    	// Evaluate model on training examples and compute training error
			// 	val valuesAndPreds = testing.map { point =>
		 //  			val prediction = model.predict(point.features)
		 //  			// val test = abs(point.label-prediction)
		 //  			(point.label, prediction)
			// 	}

			// 	val test = valuesAndPreds.map{ case(v,p) => math.abs(v-p)}
			// 	val incorrect_prediction = test.filter(label => label.toInt == 1)

			// 	// get error rate of model
			// 	val incorrect_count = incorrect_prediction.count.toDouble
			// 	val total_count = test.count.toDouble
			// 	val error_rate = (incorrect_count / total_count)*100

			// 	total_error_rate = total_error_rate + error_rate
			// 	total_pos_instances = total_pos_instances + pos_count
			// 	total_neg_instances = total_neg_instances + neg_count

			// 	println(s"Training Error Rate: $error_rate%")

			// 	if(error_rate<lowest_error_rate){
			// 		lowest_error_rate = error_rate
			// 		best_model_loop = x
			// 	}
			// }
			// else if(x==4){
			// 	val training = sc.union(split1,split2,split3,split5,split6,split7,split8,split9,split10)
			// 	val testing = split4
			// 	training.cache()
			// 	testing.cache()
			// 	val model = lrLearner1.run(training)

		 //    	// Evaluate model on training examples and compute training error
			// 	val valuesAndPreds = testing.map { point =>
		 //  			val prediction = model.predict(point.features)
		 //  			// val test = abs(point.label-prediction)
		 //  			(point.label, prediction)
			// 	}

			// 	val test = valuesAndPreds.map{ case(v,p) => math.abs(v-p)}
			// 	val incorrect_prediction = test.filter(label => label.toInt == 1)

			// 	// get error rate of model
			// 	val incorrect_count = incorrect_prediction.count.toDouble
			// 	val total_count = test.count.toDouble
			// 	val error_rate = (incorrect_count / total_count)*100

			// 	total_error_rate = total_error_rate + error_rate
			// 	total_pos_instances = total_pos_instances + pos_count
			// 	total_neg_instances = total_neg_instances + neg_count

			// 	println(s"Training Error Rate: $error_rate%")

			// 	if(error_rate<lowest_error_rate){
			// 		lowest_error_rate = error_rate
			// 		best_model_loop = x
			// 	}
			// }
			// else if(x==5){
			// 	val training = sc.union(split1,split2,split3,split4,split6,split7,split8,split9,split10)
			// 	val testing = split5
			// 	training.cache()
			// 	testing.cache()
			// 	val model = lrLearner1.run(training)

		 //    	// Evaluate model on training examples and compute training error
			// 	val valuesAndPreds = testing.map { point =>
		 //  			val prediction = model.predict(point.features)
		 //  			// val test = abs(point.label-prediction)
		 //  			(point.label, prediction)
			// 	}

			// 	val test = valuesAndPreds.map{ case(v,p) => math.abs(v-p)}
			// 	val incorrect_prediction = test.filter(label => label.toInt == 1)

			// 	// get error rate of model
			// 	val incorrect_count = incorrect_prediction.count.toDouble
			// 	val total_count = test.count.toDouble
			// 	val error_rate = (incorrect_count / total_count)*100

			// 	total_error_rate = total_error_rate + error_rate
			// 	total_pos_instances = total_pos_instances + pos_count
			// 	total_neg_instances = total_neg_instances + neg_count

			// 	println(s"Training Error Rate: $error_rate%")

			// 	if(error_rate<lowest_error_rate){
			// 		lowest_error_rate = error_rate
			// 		best_model_loop = x
			// 	}
			// }
			// else if(x==6){
			// 	val training = sc.union(split1,split2,split3,split4,split5,split7,split8,split9,split10)
			// 	val testing = split6
			// 	training.cache()
			// 	testing.cache()
			// 	val model = lrLearner1.run(training)

		 //    	// Evaluate model on training examples and compute training error
			// 	val valuesAndPreds = testing.map { point =>
		 //  			val prediction = model.predict(point.features)
		 //  			// val test = abs(point.label-prediction)
		 //  			(point.label, prediction)
			// 	}

			// 	val test = valuesAndPreds.map{ case(v,p) => math.abs(v-p)}
			// 	val incorrect_prediction = test.filter(label => label.toInt == 1)

			// 	// get error rate of model
			// 	val incorrect_count = incorrect_prediction.count.toDouble
			// 	val total_count = test.count.toDouble
			// 	val error_rate = (incorrect_count / total_count)*100

			// 	total_error_rate = total_error_rate + error_rate
			// 	total_pos_instances = total_pos_instances + pos_count
			// 	total_neg_instances = total_neg_instances + neg_count

			// 	println(s"Training Error Rate: $error_rate%")

			// 	if(error_rate<lowest_error_rate){
			// 		lowest_error_rate = error_rate
			// 		best_model_loop = x
			// 	}
			// }
			// else if(x==7){
			// 	val training = sc.union(split1,split2,split3,split4,split5,split6,split8,split9,split10)
			// 	val testing = split7
			// 	training.cache()
			// 	testing.cache()
			// 	val model = lrLearner1.run(training)

		 //    	// Evaluate model on training examples and compute training error
			// 	val valuesAndPreds = testing.map { point =>
		 //  			val prediction = model.predict(point.features)
		 //  			// val test = abs(point.label-prediction)
		 //  			(point.label, prediction)
			// 	}

			// 	val test = valuesAndPreds.map{ case(v,p) => math.abs(v-p)}
			// 	val incorrect_prediction = test.filter(label => label.toInt == 1)

			// 	// get error rate of model
			// 	val incorrect_count = incorrect_prediction.count.toDouble
			// 	val total_count = test.count.toDouble
			// 	val error_rate = (incorrect_count / total_count)*100

			// 	total_error_rate = total_error_rate + error_rate
			// 	total_pos_instances = total_pos_instances + pos_count
			// 	total_neg_instances = total_neg_instances + neg_count

			// 	println(s"Training Error Rate: $error_rate%")

			// 	if(error_rate<lowest_error_rate){
			// 		lowest_error_rate = error_rate
			// 		best_model_loop = x
			// 	}
			// }
			// else if(x==8){
			// 	val training = sc.union(split1,split2,split3,split4,split5,split6,split7,split9,split10)
			// 	val testing = split8
			// 	training.cache()
			// 	testing.cache()
			// 	val model = lrLearner1.run(training)

		 //    	// Evaluate model on training examples and compute training error
			// 	val valuesAndPreds = testing.map { point =>
		 //  			val prediction = model.predict(point.features)
		 //  			// val test = abs(point.label-prediction)
		 //  			(point.label, prediction)
			// 	}

			// 	val test = valuesAndPreds.map{ case(v,p) => math.abs(v-p)}
			// 	val incorrect_prediction = test.filter(label => label.toInt == 1)

			// 	// get error rate of model
			// 	val incorrect_count = incorrect_prediction.count.toDouble
			// 	val total_count = test.count.toDouble
			// 	val error_rate = (incorrect_count / total_count)*100

			// 	total_error_rate = total_error_rate + error_rate
			// 	total_pos_instances = total_pos_instances + pos_count
			// 	total_neg_instances = total_neg_instances + neg_count

			// 	println(s"Training Error Rate: $error_rate%")

			// 	if(error_rate<lowest_error_rate){
			// 		lowest_error_rate = error_rate
			// 		best_model_loop = x
			// 	}
			// }
			// else if(x==9){
			// 	val training = sc.union(split1,split2,split3,split4,split5,split6,split7,split8,split10)
			// 	val testing = split9
			// 	training.cache()
			// 	testing.cache()
			// 	val model = lrLearner1.run(training)

		 //    	// Evaluate model on training examples and compute training error
			// 	val valuesAndPreds = testing.map { point =>
		 //  			val prediction = model.predict(point.features)
		 //  			// val test = abs(point.label-prediction)
		 //  			(point.label, prediction)
			// 	}

			// 	val test = valuesAndPreds.map{ case(v,p) => math.abs(v-p)}
			// 	val incorrect_prediction = test.filter(label => label.toInt == 1)

			// 	// get error rate of model
			// 	val incorrect_count = incorrect_prediction.count.toDouble
			// 	val total_count = test.count.toDouble
			// 	val error_rate = (incorrect_count / total_count)*100

			// 	total_error_rate = total_error_rate + error_rate
			// 	total_pos_instances = total_pos_instances + pos_count
			// 	total_neg_instances = total_neg_instances + neg_count

			// 	println(s"Training Error Rate: $error_rate%")

			// 	if(error_rate<lowest_error_rate){
			// 		lowest_error_rate = error_rate
			// 		best_model_loop = x
			// 	}
			// }
			// else if(x==10){
			// 	val training = sc.union(split1,split2,split3,split4,split5,split6,split7,split8,split9)
			// 	val testing = split10
			// 	training.cache()
			// 	testing.cache()
			// 	val model = lrLearner1.run(training)

		 //    	// Evaluate model on training examples and compute training error
			// 	val valuesAndPreds = testing.map { point =>
		 //  			val prediction = model.predict(point.features)
		 //  			// val test = abs(point.label-prediction)
		 //  			(point.label, prediction)
			// 	}

			// 	val test = valuesAndPreds.map{ case(v,p) => math.abs(v-p)}
			// 	val incorrect_prediction = test.filter(label => label.toInt == 1)

			// 	// get error rate of model
			// 	val incorrect_count = incorrect_prediction.count.toDouble
			// 	val total_count = test.count.toDouble
			// 	val error_rate = (incorrect_count / total_count)*100

			// 	total_error_rate = total_error_rate + error_rate
			// 	total_pos_instances = total_pos_instances + pos_count
			// 	total_neg_instances = total_neg_instances + neg_count

			// 	println(s"Training Error Rate: $error_rate%")

			// 	if(error_rate<lowest_error_rate){
			// 		lowest_error_rate = error_rate
			// 		best_model_loop = x
			// 	}
			// }
			// // else {
			// // 	val title_article = testRdd.map {
		 // //        	case (inString) =>
		 // //        		val title = title_regex_testing.findFirstIn(inString).mkString("")
		 // //        		val article = article_regex.findFirstIn(inString).mkString("")
		 // //        	(title, article)
		 // //        }

		 // //        // trying to run the testing data through
		 // //    	val testFeatures = title_article.map{f => 
		 // //    		val example = tf.transform(f._2.toLowerCase.split(" "))
		 // //    		val prediction = model.predict(example)
		 // //    		(prediction, f._1)
		 // //    	}

		 // //    	val final_result = testFeatures.map(f => (f._1 +"\t"+f._2))
		 // //    	final_result.collect().foreach(println)
			// // }

	  //   }

	  	//val training = sc.union(split2,split3,split4,split5,split6,split7,split8,split9,split10)
		//val testing = split1
		//training.cache()
		//testing.cache()
		// Run the actual learning algorithm on the training data.
    	val model = lrLearner1.run(trainingData)

    	// Evaluate model on training examples and compute training error
		// val valuesAndPreds = testing.map { point =>
  // 			val prediction = model.predict(point.features)
  // 			// val test = abs(point.label-prediction)
  // 			(point.label, prediction)
		// }

		// val test = valuesAndPreds.map{ case(v,p) => math.abs(v-p)}
		// val incorrect_prediction = test.filter(label => label.toInt == 1)

		// // get error rate of model
		// val incorrect_count = incorrect_prediction.count.toDouble
		// val total_count = test.count.toDouble
		// val error_rate = (incorrect_count / total_count)*100

		// total_error_rate = total_error_rate + error_rate
		// total_pos_instances = total_pos_instances + pos_count
		// total_neg_instances = total_neg_instances + neg_count

		// println(s"Training Error Rate: $error_rate%")

		// if(error_rate<lowest_error_rate){
		// 	lowest_error_rate = error_rate
		// 	best_model_loop = x
		// }

		val title_article = testRdd.map {
        case (inString) =>
        		val title = title_regex_testing.findFirstIn(inString).mkString("")
        		val article = article_regex.findFirstIn(inString).mkString("")
        	(title, article)
        }

        // trying to run the testing data through
    	val testFeatures = title_article.map{f => 
    		val example = tf.transform(f._2.toLowerCase.split(" "))
    		val prediction = model.predict(example)
    		(prediction, f._1)
    	}

    	val final_result = testFeatures.map(f => (f._1.toInt +"\t"+f._2))
    	//final_result.collect().foreach(println)

    	final_result.saveAsTextFile("ClassificationOutput")

	 //    total_error_rate = total_error_rate/(num_loops.toDouble)

		// println(s"\nTotal positive instances: $pos_count")
	 //    println(s"Total negative instances: $neg_count")
	 //    //println(s"Lowest error rate: $lowest_error_rate%")
	 //    println(s"\nAverage error rate: $total_error_rate%")

        sc.stop()
    }
}
