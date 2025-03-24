#include <Servo.h>
#include "definitions.h"

// LFSensor more to the Left is index "0"
int LFSensor[7]={0, 0, 0, 0, 0, 0, 0};
volatile bool run = false;

volatile char mode = 0;

// PID controller variables
float Kp=50; //12
float Ki=0;
float Kd=0; //20

float error=0, P=0, I=0, D=0, PIDvalue=0;
float previousError=0, previousI=0;

// Servo's
Servo leftServo;
Servo rightServo;

// BT Module
#include <SoftwareSerial.h>
SoftwareSerial BT1(4, 6);

// timer
unsigned long startTimer;

void setup() 
{ 
  Serial.begin(9600);
  BT1.begin(9600);
  
  // line follow sensors pin setup
  pinMode(farLeftSensor, INPUT);
  pinMode(lineFollowSensor0, INPUT);
  pinMode(lineFollowSensor1, INPUT);
  pinMode(lineFollowSensor2, INPUT);
  pinMode(lineFollowSensor3, INPUT);
  pinMode(lineFollowSensor4, INPUT);
  pinMode(farRightSensor, INPUT);
  
  // servo setup
  leftServo.attach(3);
  rightServo.attach(5);

  // initial mode
  mode = STOPPED;
  run = true;

  // wait for command to start operation
  while (mode==STOPPED) {
    checkBTcmd();  // verify if a comand is received from BT remote control
    manualCmd ();

    if(Serial.available()) {
      // Read the response from command centre
      int receivedData = Serial.parseInt();
      // Serial.print("Received data from command centre: ");
      // Check if the received data is 42
      if (receivedData == 1337) {
        // command centre sent back 42, exit while loop
        Serial.println("COMMENCING OPERATION");
        mode = FOLLOWING_LINE;
        break;
      }
    }
  }
}

void loop() 
{
  readLFSsensors();
  delay(100);

  if (!run) {
    motorStop();
    while (true) {}
  } else {
    switch (mode)
    {   
      case NO_LINE:
        // Make a 180 degree turn when we run out of line
        Serial.println("NO_LINE");
        motorStop(500);
        // motorTurnTimed180(LEFT);
        motorTurnAndFindLine(LEFT);
        break;
      
      case INTERSECTION:
        Serial.println("INTERSECTION");
        run=false;
        Serial.println("INTERSECTION STOP");
        break;
        
      case RIGHT_TURN:
        Serial.println("RIGHT_TURN");
        // check first if this is not a false detection of RIGHT_TURN
        run=false;
        Serial.println("UNEXPECTED MODE AFTER RIGHT_TURN");
        break;
        
      case LEFT_TURN:
        Serial.println("LEFT_TURN");
        run=false;
        Serial.println("UNEXPECTED MODE AFTER LEFT_TURN");
        break;
    
      case FOLLOWING_LINE:
        Serial.println("FOLLOWING_LINE");
        followLine();
        break;  

      case PAUSE_LINE:
        Serial.println("PAUSE_LINE");
        motorStop(5000);

        // Send a number from Arduino to command centre
        Serial.println("PAUSE_LINE SENDING MESSAGE");
        
        Serial.println(msgStartMeasurement);
        Serial.flush();

        // Wait for a response from command centre
        while (true) {
          if(Serial.available()) {
            // Read the response from command centre
            int receivedData = Serial.parseInt();
            // Serial.print("Received data from command centre: ");
            // Check if the received data is 42
            if (receivedData == 42) {
              // command centre sent back 42, exit while loop
              Serial.println("PAUSE_LINE RECEIVED COMMAND TO PROCEED");
              runExtraCentimetre();
              break;
            }
          }
        }
        break;

      default:
        Serial.println("UNKNOWN MODE");
        run = false;
    }
  }
}