import ray
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import pandas as pd
import torch
import json
from pyspark.sql.functions import col, explode, lit, when
from datetime import datetime, timedelta
from transformers import AutoTokenizer, LlamaTokenizerFast

# Initialize Ray with 2 actors
ray.init(ignore_reinit_error=True, log_to_driver=False)

@ray.remote(num_gpus=1)
class TicketClassifier:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32, 
            device_map="auto",
            trust_remote_code=True, 
            load_in_4bit=True  # Use 4-bit quantization
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.llama_tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer") 
    
    def classify_ticket(self, ticket_text, talk_time, hold_time, after_call_work_time):
        if len(self.llama_tokenizer.encode(ticket_text))<50:
            return {k: "Not Generated" for k in [
                "Summary", "Category", "Main Issue", "Steps Taken", "Sentiment", "Urgency", 
                "Follow-Up Actions", "Repeated Issues", "Customer Loyalty Indicators", 
                "Transfer Involved", "Department Transferred To", "Issue Resolution Status", 
                "Satisfaction Score", "Call Duration Analysis", "Improvement Suggestions"
            ]}
        system_prompt = f"""
        You are an AI assistant for the customer support team of PPL Electric Utilities.
        Your role is to analyze incoming customer support tickets and provide structured and detailed information to help our team respond quickly and effectively.

        Business Context:
        - We are an electrical utilities company based out of Pennsylvania serving customers in the PA region.  
        - We handle a variety of tickets daily, ranging from power outages to billing inquiries and renewable energy discussions.
        - Quick and accurate classification is crucial for customer satisfaction and operational efficiency.

        Your tasks:
        1. **Summarize the conversation:** Provide a concise summary of the ticket, highlighting key points discussed by the customer and the agent.
        2. **Categorize the ticket:** Assign the ticket to the most appropriate category from the following categories: Classify in the following categories: (BILLING/ BUSINESS ACCOUNTS/ COLLECTIONS/ COMMERCIAL/ ENERGY/ ON TRACK/ POWER/ STANDARD OFFER/ START OR STOP OR TRANSFER SERVICE)
        3. **Identify the main issue faced by the customer:** Clearly state the primary issue or concern raised by the customer.
        4. **Detail the steps taken to resolve the issue:** List the actions taken by the agent during the call to address the customer's concern.
        5. **Sentiment Analysis:** Analyze the emotional tone of the customer throughout the conversation.
        6. **Determine Urgency:** Assess the urgency of the issue based on the conversation. Classify in the following categories: (High/ Medium/ Low).
        7. **Identify any follow-up actions:** Detail any follow-up actions required or scheduled post-conversation.
        8. **Highlight any repeated issues:** Indicate if the issue discussed has been a repeated problem for the customer. Classify in the following categories: (Yes/No).
        9. **Customer loyalty indicators:** Evaluate if there are any indications of customer loyalty or dissatisfaction expressed during the call.
        10. **Transfer involved?:** Indicate if the call was transferred to another department. Classify in the following categories: (Yes/No).
        11. **Department transferred to:** If there was a transfer, specify the department to which the call was transferred. 
        12. **Issue resolution status:** State whether the customer's issue was resolved during the call (Resolved/Unresolved).
        13. **Satisfaction score for agent call handling:** Provide a score (1-5) assessing the agent's handling of the call, where 1 is very dissatisfied and 5 is very satisfied. Classify as (1/ 2/ 3/ 4/ 5) as output based on estimated satisfaction.
        14. **Evaluate Call Duration:** Average talk time is 6 minutes per call. Given the talk time: {talk_time} minutes, hold time: {hold_time} minutes, and after-call work time: {after_call_work_time} minutes for the given ticket, analyze why the call might have taken longer than normal provided .
        15. **Suggest Improvements in Call Handling:** Based on the analysis of the conversation, agent call handling skills and outcome of the call, suggest potential improvements to enhance efficiency and customer satisfaction.

        Analyze the following customer support ticket and provide the requested information in the specified format.

        Call: {ticket_text}

        Output Format:
        {{
            "Summary": <summary>,
            "Category": <category>,
            "Main Issue": <main issue>,
            "Steps Taken": <steps taken>,
            "Sentiment": <sentiment>,
            "Urgency": <urgency>,
            "Follow-Up Actions": <follow-up actions>,
            "Repeated Issues": <repeated issues Yes or No>,
            "Customer Loyalty Indicators": <customer loyalty indicators>,
            "Transfer Involved": <yes/no>,
            "Department Transferred To": <department>,
            "Issue Resolution Status": <resolved/unresolved>,
            "Satisfaction Score": <satisfaction score>,
            "Call Duration Analysis": <call duration analysis>,
            "Improvement Suggestions": <improvement suggestions>
        }}

        Output:
        """

        try:
            response, history = self.model.chat(self.tokenizer, system_prompt)
            return json.loads(response)
        except Exception as e:
            logging.error(f"Error in classifying ticket: {str(e)}")
            return {k: "Not Generated" for k in [
                "Summary", "Category", "Main Issue", "Steps Taken", "Sentiment", "Urgency", 
                "Follow-Up Actions", "Repeated Issues", "Customer Loyalty Indicators", 
                "Transfer Involved", "Department Transferred To", "Issue Resolution Status", 
                "Satisfaction Score", "Call Duration Analysis", "Improvement Suggestions"
            ]}
    
    def process_batch(self, batch):
        results = []
        for _, row in batch.iterrows():
            result = self.classify_ticket(
                row['transcription'],
                row['talkTime'] / 60,
                row['holdTime'] / 60,
                row['acwTime'] / 60
            )
            results.append(result)
        return results        

def parallelize_inference(df):
    BATCH_SIZE = 10  # Optimal batch size for efficient processing
    batches = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    num_actors = 2  # Two actors available
    processors = [TicketClassifier.remote("internlm/internlm2_5-7b-chat") for _ in range(num_actors)]
    futures = [processors[i % num_actors].process_batch.remote(batch) for i, batch in enumerate(batches)]
    results = ray.get(futures)
    return [item for sublist in results for item in sublist]  # Flatten the list of results

#date_reqd = "2024-08-01" 
date_reqd = datetime.now().date() - timedelta(days=4)

# Database and table details
database_name = "default"
table_name = "master_production"

# Read table into a DataFrame
transcription = spark.table(f"{database_name}.{table_name}")

transcription.display()

transcription = transcription.filter(transcription['Date'] == lit(date_reqd))

transcription = transcription.toPandas()

print("Number of Rows: "+str(len(transcription)))

result_list = parallelize_inference(transcription)
print(result_list)

# Convert result_list to DataFrame
output_df = pd.DataFrame(result_list)

# Merge with original transcription DataFrame
output_df['callSid'] = transcription['callSid']
final = transcription.merge(output_df, 
                            on=['callSid'], 
                            how='inner')

output_path = f'gpu_adv_ins_{date_reqd.strftime("%Y_%m_%d")}.csv'
final.to_csv(output_path, index=False)
print(f"Output saved to {output_path}")

final_reqd = final[['Date','callSid','Summary', 'Category', 'Main Issue', 'Steps Taken', 'Sentiment',
       'Urgency', 'Follow-Up Actions', 'Repeated Issues',
       'Customer Loyalty Indicators', 'Transfer Involved',
       'Department Transferred To', 'Issue Resolution Status',
       'Satisfaction Score', 'Call Duration Analysis',
       'Improvement Suggestions']]

final_reqd[final_reqd.columns] = final_reqd[final_reqd.columns].astype(str)
final_reqd['Date'] = final_reqd['Date'].apply(lambda x: pd.to_datetime(x).date())

#final['Satisfaction Score'] = final['Satisfaction Score'].apply(satisfaction_score_parser)
output_path = f'gpu_adv_ins_{date_reqd.strftime("%Y_%m_%d")}.csv'
final.to_csv(output_path, index=False)
print(f"Output saved to {output_path}")

data_sdf = spark.createDataFrame(final_reqd)

for column in data_sdf.columns:
    data_sdf = data_sdf.withColumnRenamed(column, column.replace(" ", "_"))

inference_date = datetime.now().date().strftime("%Y-%m-%d")
database_name = "default"
table_name = "advanced_insights"
data_sdf = data_sdf.withColumn("inference_date",lit(inference_date))

#data_sdf.write.mode("overwrite").option("overwriteSchema", "True").saveAsTable(f"{table_name}")
data_sdf.write.mode("append").option("mergeSchema", "True").saveAsTable(f"{table_name}")

ray.shutdown()
