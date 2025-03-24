#define STOPPED 0
#define FOLLOWING_LINE 1
#define NO_LINE 2
#define INTERSECTION 3
// #define POS_LINE 4
#define RIGHT_TURN 5
#define LEFT_TURN 6
#define PAUSE_LINE 7

#define RIGHT -1
#define LEFT 1

// Line sensor pin configuration
const int farLeftSensor = 2; // No longer an Analog Pin
const int lineFollowSensor0 = 12; 
const int lineFollowSensor1 = 11; 
const int lineFollowSensor2 = 10; 
const int lineFollowSensor3 = 9;
const int lineFollowSensor4 = 8;
const int farRightSensor = 7; // No longer an Analog Pin

// Line - Void colour
const int line_reflectance=0;
const int void_reflectance=1;

// motor control constants
const int power = 200; // 125
const int iniMotorPower = 200; // 125
const int adjL = 0; // 100
const int adjR = 0; // 100
const float adjTurn = 950; // time factor for turns
const float adjTurn180 = 880; // time factor for 180 degree turns
const int extraInch = 1000;
const int extraCentimetre = 1300; // 1500
const int untilAxis = 2500;
const unsigned long carLength = 6900; //8000

const int msgStartMeasurement = 18;