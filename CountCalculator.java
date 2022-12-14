// Script task Hadoop MapReduce - jumlah data STATUS

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import java.io.IOException;

public class CountCalculator {

    //Driver Class
    public static void main(String[] args) throws Exception {
        //set up configurations
        Configuration c = new Configuration();
        String[] files = new GenericOptionsParser(c, args).getRemainingArgs();
        Path input = new Path(files[0]);
        Path output = new Path(files[1]);
        Job j = new Job(c, "countCalculator");
        j.setJarByClass(CountCalculator.class);
        j.setMapperClass(MapForCount.class);
        j.setReducerClass(ReduceForCount.class);
        j.setOutputKeyClass(Text.class);
        j.setOutputValueClass(FloatWritable.class);

        //get input paths from aruguments
        FileInputFormat.addInputPath(j, input);
        FileOutputFormat.setOutputPath(j, output);

        //start measuring the start time.
        long startTime = System.currentTimeMillis();
        j.waitForCompletion(true);
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("Time Elapsed : " + estimatedTime);
        System.exit(0);
    }


    //Mapper
    public static class MapForCount extends Mapper<LongWritable, Text, Text, FloatWritable> {

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] words = line.split(",");
            Text outputKey = new Text(words[0].toUpperCase().trim());
            FloatWritable outputValue = new FloatWritable(Float.parseFloat(words[6]));
            context.write(outputKey, outputValue);
        }
    }

    //Reducer
    public static class ReduceForCount extends Reducer<Text, FloatWritable, Text, FloatWritable> {

        public void reduce(Text word, Iterable<FloatWritable> values, Context context) throws IOException, InterruptedException {
            float count = 0;
            for (FloatWritable value : values) {
                count = count + 1;
            }

            context.write(word, new FloatWritable(count));
        }
    }
}