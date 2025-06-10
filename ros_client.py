import json
import os
import time
import roslibpy


def main():
    ros = roslibpy.Ros(host=os.getenv('ROSBRIDGE_HOST', 'localhost'),
                       port=int(os.getenv('ROSBRIDGE_PORT', '9090')))
    ros.run()

    request = {
        'rows': 5,
        'cols': 5,
        'start': [0, 0],
        'obstacles': [],
        'orientation': 0,
        'heuristic': 'vertical'
    }

    pub = roslibpy.Topic(ros, '/getPlan', 'nav_msgs/GetPlan')
    sub = roslibpy.Topic(ros, '/planned_grid', 'nav_msgs/OccupancyGrid')

    received = {'flag': False}

    def handle_grid(message):
        print('Received planned grid:')
        print(json.dumps(message, indent=2))
        received['flag'] = True

    sub.subscribe(handle_grid)

    pub.publish(roslibpy.Message(request))
    print('Request sent, waiting for planned grid...')

    try:
        while ros.is_connected and not received['flag']:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sub.unsubscribe()
        pub.unadvertise()
        ros.terminate()


if __name__ == '__main__':
    main()
