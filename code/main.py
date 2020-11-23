from argparse import ArgumentParser
import os
import task_listener
from utils.ip_provider import get_valid_ip


def main():
    parser = ArgumentParser()
    parser.add_argument("-bs", "--batch_size",dest="batch_size", default=256,type=int)
    parser.add_argument("-e","--epochs", dest="epochs", default=25,type=int)
    parser.add_argument("-ife", "--is_feature_extraction", dest="is_feature_extraction", default=1,type=int,help="0=False,1=True")
    parser.add_argument("-d", "--data_dir", dest="data_dir", default=os.path.join("..","shared","data"))
    parser.add_argument("-r", "--results_dir", dest="results_dir", default=os.path.join("..","shared","results"))
    parser.add_argument("-rmqs", "--rabbitmq_server", dest="rabbitmq_server", default="messagebroker") #sollte durch den k8s proxy aufgelöst werden.
    parser.add_argument("-dbh", "--database_host", dest="database_host", default="postgres") #sollte durch den k8s proxy aufgelöst werden.
    parser.add_argument("-db", "--database", dest="database", default="elfi_elite_final_20200827")
    parser.add_argument("-dbu", "--database_user", dest="database_user", default="user")
    parser.add_argument("-dbpw", "--database_password", dest="database_password", default="password123")
    parser.add_argument("-eit", "--earlystopping_it", dest="earlystopping_it", default=5,type=int)
    parser.add_argument("-test_raxtn", "--test_results_ax_table_name", dest="test_results_ax_table_name", default="test_results_ax")
    parser.add_argument("-vraxtn", "--validation_results_ax_table_name", dest="validation_results_ax_table_name", default="validation_results_ax")
    parser.add_argument("-traxtn", "--train_results_ax_table_name", dest="train_results_ax_table_name", default="train_results_ax")
    parser.add_argument("-rd", "--run_dir", dest="run_dir", default="candar_runs")
    parser.add_argument("-ic", "--internet_conn_exist", dest="internet_conn_exist", default=0, type=int, help="0=False,1=True")
    args = parser.parse_args()
    
    args.rabbitmq_server = get_valid_ip(args.rabbitmq_server)
    args.database_host = get_valid_ip(args.database_host)

    task_listener.start_task_listener(args)

if __name__ == "__main__":
    main()
