import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql
import org.apache.spark.sql.functions.{col, sum, _}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object Airbnb {
  def evaluateDataQuality(df: DataFrame) {
    df.printSchema()
    //    root
    //    |-- id: integer (nullable = true)
    //    |-- bathrooms: double (nullable = true)
    //    |-- bed_type: string (nullable = true)
    //    |-- bedrooms: integer (nullable = true)
    //    |-- beds: integer (nullable = true)
    //    |-- cancellation_policy: string (nullable = true)
    //    |-- latitude: double (nullable = true)
    //    |-- longitude: double (nullable = true)
    //    |-- neighbourhood_group_cleansed: string (nullable = true)
    //    |-- price: integer (nullable = true)
    //    |-- property_type: string (nullable = true)
    //    |-- room_type: string (nullable = true)
    //    |-- "Carbon monoxide detector": boolean (nullable = false)
    //    |-- "Cable TV": boolean (nullable = false)
    //    |-- "Self check-in": boolean (nullable = false)
    //    |-- "Cooking basics": boolean (nullable = false)
    //    |-- "Luggage dropoff allowed": boolean (nullable = false)
    //    |-- "Private entrance": boolean (nullable = false)
    //    |-- "Host greets you": boolean (nullable = false)
    //    |-- "Dishes and silverware": boolean (nullable = false)
    //    |-- Stove: boolean (nullable = false)
    //    |-- Internet: boolean (nullable = false)
    //    |-- "Bed linens": boolean (nullable = false)
    //    |-- "First aid kit": boolean (nullable = false)
    //    |-- "No stairs or steps to enter": boolean (nullable = false)
    //    |-- Microwave: boolean (nullable = false)
    //    |-- Heating: boolean (nullable = false)
    //    |-- "Family/kid friendly": boolean (nullable = false)
    //    |-- Refrigerator: boolean (nullable = false)
    //    |-- "Fire extinguisher": boolean (nullable = false)
    //    |-- "Free parking on premises": boolean (nullable = false)
    //    |-- "Smoke detector": boolean (nullable = false)
    //    |-- "Long term stays allowed": boolean (nullable = false)
    //    |-- Gym: boolean (nullable = false)
    //    |-- Pool: boolean (nullable = false)
    //    |-- "Hot water": boolean (nullable = false)
    //    |-- "Lock on bedroom door": boolean (nullable = false)
    //    |-- Dryer: boolean (nullable = false)
    //    |-- "Laptop friendly workspace": boolean (nullable = false)
    //    |-- "Hair dryer": boolean (nullable = false)
    //    |-- Elevator: boolean (nullable = false)
    //    |-- Iron: boolean (nullable = false)
    //    |-- Shampoo: boolean (nullable = false)
    //    |-- TV: boolean (nullable = false)
    //    |-- Hangers: boolean (nullable = false)
    //    |-- Kitchen: boolean (nullable = false)
    //    |-- Essentials: boolean (nullable = false)
    //    |-- Washer: boolean (nullable = false)
    //    |-- Wifi: boolean (nullable = false)
    //    |-- "Air conditioning": boolean (nullable = false)
    df.describe().show()

    def check_missing(df: DataFrame): Unit = {
      df.select(df.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show
    }

    // Check missing again
    println("How many missing per column?")
    check_missing(df)
    //    +---+---------+--------+--------+----+-------------------+--------+---------+----------------------------+-----+-------------+---------+--------------------------+----------+---------------+----------------+-------------------------+------------------+-----------------+-----------------------+-----+--------+------------+---------------+-----------------------------+---------+-------+---------------------+------------+-------------------+--------------------------+----------------+-------------------------+---+----+-----------+----------------------+-----+---------------------------+------------+--------+----+-------+---+-------+-------+----------+------+----+------------------+
    //    | id|bathrooms|bed_type|bedrooms|beds|cancellation_policy|latitude|longitude|neighbourhood_group_cleansed|price|property_type|room_type|"Carbon monoxide detector"|"Cable TV"|"Self check-in"|"Cooking basics"|"Luggage dropoff allowed"|"Private entrance"|"Host greets you"|"Dishes and silverware"|Stove|Internet|"Bed linens"|"First aid kit"|"No stairs or steps to enter"|Microwave|Heating|"Family/kid friendly"|Refrigerator|"Fire extinguisher"|"Free parking on premises"|"Smoke detector"|"Long term stays allowed"|Gym|Pool|"Hot water"|"Lock on bedroom door"|Dryer|"Laptop friendly workspace"|"Hair dryer"|Elevator|Iron|Shampoo| TV|Hangers|Kitchen|Essentials|Washer|Wifi|"Air conditioning"|
    //    +---+---------+--------+--------+----+-------------------+--------+---------+----------------------------+-----+-------------+---------+--------------------------+----------+---------------+----------------+-------------------------+------------------+-----------------+-----------------------+-----+--------+------------+---------------+-----------------------------+---------+-------+---------------------+------------+-------------------+--------------------------+----------------+-------------------------+---+----+-----------+----------------------+-----+---------------------------+------------+--------+----+-------+---+-------+-------+----------+------+----+------------------+
    //    |  0|        0|       0|       0|   0|                  0|       0|        0|                           0|    0|            0|        0|                         0|         0|              0|               0|                        0|                 0|                0|                      0|    0|       0|           0|              0|                            0|        0|      0|                    0|           0|                  0|                         0|               0|                        0|  0|   0|          0|                     0|    0|                          0|           0|       0|   0|      0|  0|      0|      0|         0|     0|   0|                 0|
    //    +---+---------+--------+--------+----+-------------------+--------+---------+----------------------------+-----+-------------+---------+--------------------------+----------+---------------+----------------+-------------------------+------------------+-----------------+-----------------------+-----+--------+------------+---------------+-----------------------------+---------+-------+---------------------+------------+-------------------+--------------------------+----------------+-------------------------+---+----+-----------+----------------------+-----+---------------------------+------------+--------+----+-------+---+-------+-------+----------+------+----+------------------+

    println("Summary of data frame:")
    df.describe().show()
    //    +-------+--------------------+------------------+--------+------------------+------------------+-------------------+--------------------+------------------+----------------------------+------------------+-------------+---------------+
    //    |summary|                  id|         bathrooms|bed_type|          bedrooms|              beds|cancellation_policy|            latitude|         longitude|neighbourhood_group_cleansed|             price|property_type|      room_type|
    //    +-------+--------------------+------------------+--------+------------------+------------------+-------------------+--------------------+------------------+----------------------------+------------------+-------------+---------------+
    //    |  count|                7840|              7840|    7840|              7840|              7840|               7840|                7840|              7840|                        7840|              7840|         7840|           7840|
    //    |   mean|2.3406948718367346E7| 1.550063775510204|    null| 1.336734693877551|2.0096938775510202|               null|  1.3142366186224475|103.84884752678549|                        null|150.69630102040816|         null|           null|
    //    | stddev|1.0142627875734216E7|1.2922092779328715|    null|1.1148500916711868| 2.223833353283169|               null|0.030510639843313585|0.0436304967117557|                        null| 119.4230677142169|         null|           null|
    //    |    min|               49091|               0.0|  Airbed|                 0|                 0|           flexible|             1.24387|         103.66547|              Central Region|                 0|   Aparthotel|Entire home/apt|
    //    |    max|            38112762|              50.0|Real Bed|                50|                36|    super_strict_60|             1.45459|         103.97342|                 West Region|               999|        Villa|    Shared room|
    //    +-------+--------------------+------------------+--------+------------------+------------------+-------------------+--------------------+------------------+----------------------------+------------------+-------------+---------------+

    df.select("bed_type").distinct().collect().foreach(println)
    //      [Airbed]
    //      [Futon]
    //      [Pull-out Sofa]
    //      [Couch]
    //      [Real Bed]
    df.select("cancellation_policy").distinct().collect().foreach(println)
    //      [flexible]
    //      [super_strict_60]
    //      [strict]
    //      [super_strict_30]
    //      [moderate]
    //      [strict_14_with_grace_period]
    df.select("neighbourhood_group_cleansed").distinct().collect().foreach(println)
    //      [West Region]
    //      [Central Region]
    //      [North Region]
    //      [East Region]
    //      [North-East Region]
    df.select("property_type").distinct().collect().foreach(println)
    //      [Apartment]
    //      [Townhouse]
    //      [Farm stay]
    //      [Guest suite]
    //      [Boutique hotel]
    //      [Castle]
    //      [Loft]
    //      [Guesthouse]
    //      [Hostel]
    //      [Villa]
    //      [Campsite]
    //      [Aparthotel]
    //      [Other]
    //      [Serviced apartment]
    //      [Hotel]
    //      [Condominium]
    //      [House]
    //      [Chalet]
    //      [Bus]
    //      [Tent]
    //      [Boat]
    //      [Bungalow]
    //      [Bed and breakfast]
    //      [Cabin]
    df.select("room_type").distinct().collect().foreach(println)
    //      [Shared room]
    //      [Entire home/apt]
    //      [Private room]


    // Above results seems weird.
    // [Bed and breakfast]


    // Check if outliers exist
    df.filter("bathrooms == 0").show()
    df.filter("bathrooms == 50").show()
    df.filter("bathrooms == 50").show()
    df.filter("beds == 36").show() // somehow make sense, leave the data alone

    // Check duplication
    println("Is there any duplication?\n" + (df.count() - df.distinct().count())) // no duplications
  }

  def dataExplore(spark: SparkSession, airbnbDF: DataFrame): DataFrame = {
    val cols = Seq("amenities", "id", "bathrooms", "bed_type", "bedrooms", "beds", "cancellation_policy",
      "latitude", "longitude", "neighbourhood_group_cleansed", "price", "property_type", "room_type")

    // Create a new data frame only containing above cols from AirbnbDF
    var myAirbnbDF = airbnbDF.select(cols.head, cols.tail: _*)

    // Find out the most common amenities in all listings
    val amenities: Array[Row] = myAirbnbDF.select("amenities").distinct.collect()
    var amenitiesList: List[String] = List()
    for (amenity <- amenities) {
      val str = amenity(0).toString
      if (str.length - 1 >= 1) {
        // str e.g.  "xxxxxxxx, yyyyyy, zzzzz"
        amenitiesList = amenitiesList union (str.trim.substring(1, str.length - 1).split(',').toList)
      }
    }

    // Create frequency list
    val frequent_amenities: List[String] = amenitiesList
      .map(word => (word, 1))
      .groupBy(_._1)
      .mapValues(_.map(_._2).sum)
      .toList.sortBy(_._2)
      .filter(_._2 >= 1000)
      .map(word => word._1)

    // Replace amenities column with multiple columns from frequency list
    for (amenity <- frequent_amenities) {
      val amenity_striped = amenity.trim.replaceAll(" ", "_").replaceAll("\"", "")
      myAirbnbDF = myAirbnbDF.select(
        col("*"),
        col("amenities").contains(amenity).as(amenity_striped)
      )
      myAirbnbDF = myAirbnbDF.na.fill(Map(amenity_striped -> false))
    }

    // Transformation
    myAirbnbDF = myAirbnbDF
      .drop("amenities")
      // Change column types
      // to Integer
      .withColumn("id", myAirbnbDF.col("id").cast(sql.types.IntegerType))
      .withColumn("bedrooms", myAirbnbDF.col("bedrooms").cast(sql.types.IntegerType))
      .withColumn("beds", myAirbnbDF.col("beds").cast(sql.types.IntegerType))
      // to double
      .withColumn("bathrooms", myAirbnbDF.col("bathrooms").cast(sql.types.DoubleType))
      .withColumn("latitude", myAirbnbDF.col("latitude").cast(sql.types.DoubleType))
      .withColumn("longitude", myAirbnbDF.col("longitude").cast(sql.types.DoubleType))

    // Price column: dollar to number
    myAirbnbDF = myAirbnbDF.withColumn("price", substring_index(col("price"), "$", -1))
    myAirbnbDF = myAirbnbDF
      .withColumn("price", myAirbnbDF.col("price").cast(sql.types.IntegerType))
      // bathrooms, bedrooms, beds, latitude, longitude have several missing values.
      // drop missing values
      .filter(myAirbnbDF.col("bathrooms").isNotNull)
      .filter(myAirbnbDF.col("bedrooms").isNotNull)
      .filter(myAirbnbDF.col("beds").isNotNull)
      .filter(myAirbnbDF.col("latitude").isNotNull)
      .filter(myAirbnbDF.col("longitude").isNotNull)
      .filter(myAirbnbDF.col("price").isNotNull)
      // From observation, only 1 daily price of 0. remove this extreme outlier for dataset
      .filter("price != 0")

    // one-hot encoding for categorical data
    val stringCols = Array("bed_type", "cancellation_policy", "neighbourhood_group_cleansed", "property_type", "room_type")

    val indexers = stringCols.map {
      colName => new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed")
    }

    val pipeline = new Pipeline().setStages(indexers)

    val myAirbnbDFIndexed = pipeline.fit(myAirbnbDF).transform(myAirbnbDF)

    val stringColsIndexed = Array("bed_type_indexed", "cancellation_policy_indexed",
      "neighbourhood_group_cleansed_indexed", "property_type_indexed", "room_type_indexed")
    val stringColsVector = Array("bed_type_vector", "cancellation_policy_vector",
      "neighbourhood_group_cleansed_vector", "property_type_vector", "room_type_vector")

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(stringColsIndexed)
      .setOutputCols(stringColsVector)

    val model = encoder.fit(myAirbnbDFIndexed)

    val myAirbnbDFEncoded = model.transform(myAirbnbDFIndexed)

    return myAirbnbDFEncoded
  }

  def dataModeling(spark: SparkSession, dfAirbnb: DataFrame): Unit = {

    val inputCols = Array("bathrooms", "bedrooms", "beds", "latitude", "longitude",  //original numerical values
      //one hot encoding
      "bed_type_vector", "cancellation_policy_vector", "neighbourhood_group_cleansed_vector", "property_type_vector", "room_type_vector",
      //derived booleans from amenities
      "Carbon_monoxide_detector", "Cable_TV", "Self_check-in", "Cooking_basics", "Luggage_dropoff_allowed",
      "Private_entrance", "Host_greets_you", "Dishes_and_silverware", "Stove", "Internet", "Bed_linens",
      "First_aid_kit", "No_stairs_or_steps_to_enter", "Microwave", "Heating", "Family/kid_friendly",
      "Refrigerator", "Fire_extinguisher", "Free_parking_on_premises", "Smoke_detector",
      "Long_term_stays_allowed", "Gym", "Pool", "Hot_water", "Lock_on_bedroom_door", "Dryer",
      "Laptop_friendly_workspace", "Hair_dryer", "Elevator", "Iron", "Shampoo", "TV", "Hangers",
      "Kitchen", "Essentials", "Washer", "Wifi", "Air_conditioning", "price")

    val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features")

    val featureDF = assembler.transform(dfAirbnb)

    val Array(training, test) = featureDF.randomSplit(Array[Double](0.8, 0.2), 18)

    val k = 37
    val kmeansEstimator = new KMeans().setK(k).setSeed(1L).setFeaturesCol("features").setPredictionCol("prediction")
    val kmeansModel = kmeansEstimator.fit(training)

    // Make predictions
    val predictDf = kmeansModel.transform(test)

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictDf)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // Shows the result.
    println("Cluster Centers: ")
    kmeansModel.clusterCenters.foreach(println)

    val resultDF = kmeansModel.transform(featureDF)
    resultDF.groupBy("prediction").count().show(k)
    resultDF.groupBy("prediction").agg(mean("price")).show(k)

  }

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("Airbnb-Kmeans")
      .config("spark.master", "local")
      .getOrCreate()

    val airbnbDF = spark.read.option("header", "true")
      .option("escape", "\"")
      .option("multiLine", "true")
      .csv("data/ABB_listings.csv")

    val dfClean = dataExplore(spark, airbnbDF)

    evaluateDataQuality(dfClean)

    dataModeling(spark, dfClean)
  }
}
