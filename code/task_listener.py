import pika
import psycopg2
from utils.messagebroker_new_tasks import create_new_tasks
from utils.consts import model_dict
from BOBO_hypterparameter_search import hyperparameter_optimization

    
def start_task_listener(args):

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=args.rabbitmq_server,
            heartbeat=0 
        )
    )

    channel = connection.channel()

    channel.queue_declare(queue="task_queue_ho",durable=True)

    print(" [*] Waiting for tasks. To exit press CTRL+C")

    def callback(ch,method,properties,body):
        task = body.decode("utf-8")
        print(" [x] Received " + task)
        try:
            conn = psycopg2.connect(host=args.database_host,database=args.database,user=args.database_user,password=args.database_password)
        except Exception as e:
            raise e
        if task.split(":")[0] == "HO":
            finished_successfully = hyperparameter_optimization(args, conn,task)
        conn.close()            
        print(" [x] Done |:->")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)#this is important to prefetch only one task <- heavily influence the way the tasks are spread over the cluster

    channel.basic_consume(queue="task_queue_ho",on_message_callback=callback)

    channel.start_consuming()
