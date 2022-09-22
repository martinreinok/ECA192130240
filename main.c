#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

/*DATA STRUCT*/
struct Measurement{
    int distance;
    int angle;
};

/*PIN DEFINES*/
#define TRIG_PIN 1		// Trigger pin at GPIO 1 (pin 9) (OUTPUT)
#define ECHO_PIN 9		// Echo pin at GPIO 9 (pin 15) (INPUT)
#define SERVO_PIN 11	// Servo pin at GPIO 11 (pin 17) (OUTPUT)

/*FUNCTION DECLARATIONS*/
void setInputPin(uint32_t pinNumber);
void setOutputPin(uint32_t pinNumber);
void setPWMPin(uint32_t pinNumber);
void rtc_setup();
uint32_t get_rtc_low();
uint32_t get_rtc_low_micro();
uint32_t get_rtc_low_milli();
uint32_t get_rtc_high();
void initSensor();
void setPWM1(uint32_t frequency, uint32_t dutyCycle);
void setDutyCycle1(uint32_t dutyCycle);
void resetSingleShot();
void setPWM2(uint32_t frequency, uint32_t dutyCycle);
void setDutyCycle2(uint32_t dutyCycle);
void setSingleShotPWM(uint32_t pinNumber, uint32_t frequency, uint32_t dutyCycle);
void setContinuousPWM(uint32_t pinNumber, uint32_t frequency, uint32_t dutyCycle);
void save_measurement(int distance, int angle, uint32_t measurement);
void streamMeasurements();
void stream_output(uint32_t distance, uint32_t angle);
uint32_t read_echo();
float calcMotorAngle(uint32_t curposition, bool curDirection);
uint32_t calcNewPosition(uint32_t curposition, bool curDirection);

/*Global DECLARATIONS/VARIABLES*/
#define MeasurementAmount 65
#define sweepAmount 3
#define amountClosest 3
#define ServoPWMFreq 200
#define minDutyCycle 10000 		// 10%
#define maxDutyCycle 37950		// 37.95%
#define maxServoAngle 130
#define triggerDutyCycle 50000 // 50%
#define triggerPWMFreq 50000
static uint32_t BASE_FREQUENCY = 16000000;
struct Measurement closestPositions [MeasurementAmount] = {}; // All the closest positions measured
struct Measurement closestMeasurements[amountClosest] = {}; // 3 Closest positions measured
static uint32_t speed_of_sound = 343; // m/s
static uint32_t sensor_timeout = 25000;  // Units ?
static uint32_t loopTime = 60000;
static uint64_t clock_frequency = 32768;
static uint64_t clock_multiplier_micro = 1000000;
static uint64_t clock_multiplier_milli = 1000;
bool sweepDirection = false;		// Direction of sweep

/*REGISTRY DECLARATIONS*/
volatile uint32_t *GPIO_INPUT_VAL 	= (uint32_t*)0x10012000;		// Registry for INPUT values
volatile uint32_t *GPIO_INPUT_EN 	= (uint32_t*)0x10012004;		// Registry for INPUT set pin ENABLE
volatile uint32_t *GPIO_OUTPUT_VAL 	= (uint32_t*)0x1001200C;		// Registry for OUTPUT values
volatile uint32_t *GPIO_OUTPUT_EN 	= (uint32_t*)0x10012008;		// Registry for OUTPUT set pin ENALBE
volatile uint32_t *GPIO_PUE 		= (uint32_t*)0x10012010;		// Registry for INPUT Internal PULL UP ENABLE
volatile uint32_t *IOF_EN 			= (uint32_t*)0x10012038;		// Registry for HW-Driven funtions (IOF)
volatile uint32_t *IOF_SEL 			= (uint32_t*)0x1001203C;		// Registry for HW-Driven funtions (IOF)

volatile uint32_t *RTC_OUTPUT_HIGH 	= (uint32_t*)0x1000004C; 		// High bits of RTC Counter
volatile uint32_t *RTC_OUTPUT_LOW 	= (uint32_t*)0x10000048; 		// Low bits of RTC Counter
volatile uint32_t *RTC_CONFIG		= (uint32_t*)0x10000040; 		// Low bits of RTC Counter

volatile uint32_t *PWM0_1_CONFIG	= (uint32_t*)0x10015000;		// Registry for PWM CONFIG TRIGGER PIN
volatile uint32_t *PWM0_1_COUNT		= (uint32_t*)0x10015008;		// Registry for PWM Counter TRIGGER PIN
volatile uint32_t *PWM0_1_CMP0		= (uint32_t*)0x10015020;		// Registry for PWM COMPERATOR TRIGGER PIN
volatile uint32_t *PWM0_1_CMP1		= (uint32_t*)0x10015024;		// Registry for PWM COMPERATOR TRIGGER PIN
volatile uint32_t *PWM0_1_CMP2		= (uint32_t*)0x10015028;		// Registry for PWM COMPERATOR TRIGGER PIN
volatile uint32_t *PWM0_1_CMP3		= (uint32_t*)0x1001502C;		// Registry for PWM COMPERATOR TRIGGER PIN

volatile uint32_t *PWM2_1_CONFIG	= (uint32_t*)0x10035000;		// Registry for PWM CONFIG SERVO PIN
volatile uint32_t *PWM2_1_COUNT		= (uint32_t*)0x10035008;		// Registry for PWM Counter SERVO PIN
volatile uint32_t *PWM2_1_CMP0		= (uint32_t*)0x10035020;		// Registry for PWM COMPERATOR SERVO PIN
volatile uint32_t *PWM2_1_CMP1		= (uint32_t*)0x10035024;		// Registry for PWM COMPERATOR SERVO PIN
volatile uint32_t *PWM2_1_CMP2		= (uint32_t*)0x10035028;		// Registry for PWM COMPERATOR SERVO PIN
volatile uint32_t *PWM2_1_CMP3		= (uint32_t*)0x1003502C;		// Registry for PWM COMPERATOR SERVO PIN


int main() {

	setInputPin(ECHO_PIN);

	// Initialize One shot pwm at Trigger pin
	setSingleShotPWM(TRIG_PIN, triggerPWMFreq, triggerDutyCycle);
	// Initialize servo
	// Move servo to starting position 1 / 65
	setContinuousPWM(SERVO_PIN, ServoPWMFreq, minDutyCycle);	// Minimum position of rotation.

	rtc_setup();

	initSensor(); // Stabilizes the sensor output

	for(uint32_t sweep = 1; sweep <= sweepAmount; sweep ++){

		for(uint32_t measurement = 1; measurement <= MeasurementAmount; measurement ++){
			uint32_t motor_angle = calcMotorAngle(measurement, sweepDirection);

			uint32_t distance = 0;

			// Take an average of 3 measurements
			for(uint32_t i = 1; i <= 3; i ++){
				uint32_t minLoopTime = get_rtc_low_micro() + loopTime;	// Minimum time for loop to complete
				// Trigger sensor
				resetSingleShot();
				// Measure the distance;
				distance += (read_echo() * speed_of_sound)/2000;

				while(get_rtc_low_micro() <= minLoopTime); 			// wait for minimum looptime
			}

			// Calculate the average distance
			distance = distance / 3;
			// Change timeout distance to maximum sensing distance
			if(distance == 0){
				distance = 4000; // Maximum sensing distance of sensor
			}

			printf("\r \n");
			printf("\r Current measurement: %u \n", measurement);

			// Print results in terminal
			stream_output(distance, motor_angle);
			// Save measurements in array
			if(sweepDirection == false){
				save_measurement(distance, motor_angle, measurement);
			}
			else if(sweepDirection == true){
				save_measurement(distance, motor_angle, MeasurementAmount - measurement);
			}

			printf("\r First: %d (%d), Second: %d (%d), Third: %d (%d) \n", closestMeasurements[0].distance, closestMeasurements[0].angle, closestMeasurements[1].distance, closestMeasurements[1].angle, closestMeasurements[2].distance, closestMeasurements[2].angle);

			setDutyCycle2(calcNewPosition(measurement, sweepDirection));
		}

		sweepDirection = !sweepDirection;

	}

	streamMeasurements();

	/*
	// Code for testing the sensors accuracy
	initSensor(); // Stabilizes the sensor output
	uint32_t i;
	uint32_t distance = 0;
	uint32_t distanceSum = 0;
	for(i = 1; i <= 10; i ++){
		uint32_t minLoopTime = get_rtc_low_micro() + loopTime;	// Minimum time for loop to complete
		// Trigger sensor
		resetSingleShot();
		// Measure the distance;
		distance = (read_echo() * speed_of_sound)/2000;
		distanceSum += distance;
		printf("\r Measurement: %u \n", i);
		printf("\r Distance: %u \n", distance);
		while(get_rtc_low_micro() <= minLoopTime); 			// wait for minimum looptime
	}
	printf("\r Distance sum: %u \n", distanceSum);
	*/

	/*
	// Code for testing min-center-max angles of servo
	while(1) {
		uint32_t waitTime = get_rtc_low_micro() + 2000000;	// Minimum time for loop to complete
		while(get_rtc_low_micro() <= waitTime);
		setDutyCycle2(minDutyCycle);	// Minimum position of rotation.
		waitTime = get_rtc_low_micro() + 2000000;	// Minimum time for loop to complete
		while(get_rtc_low_micro() <= waitTime);
		setDutyCycle2(23975);	// Middle position of rotation.
		waitTime = get_rtc_low_micro() + 2000000;	// Minimum time for loop to complete
		while(get_rtc_low_micro() <= waitTime);
		setDutyCycle2(maxDutyCycle);	// Maximum position of rotation.
	}
	*/
}

void setInputPin(uint32_t pinNumber){
	*IOF_EN 		&= ~(1<<pinNumber);
	*GPIO_OUTPUT_EN &= ~(1<<pinNumber);
	*GPIO_INPUT_EN 	|= (1<<pinNumber);
	*GPIO_PUE 		|= (1<<pinNumber);
}

void setOutputPin(uint32_t pinNumber){
	*IOF_EN 		&= ~(1<<pinNumber);
	*GPIO_INPUT_EN 	&= ~(1<<pinNumber);
	*GPIO_PUE 		&= ~(1<<pinNumber);
	*GPIO_OUTPUT_EN |= (1<<pinNumber);

}

void setPWMPin(uint32_t pinNumber){
	*GPIO_INPUT_EN 	&= ~(1<<pinNumber);
	*GPIO_PUE 		&= ~(1<<pinNumber);
	*GPIO_OUTPUT_EN &= ~(1<<pinNumber);
	*GPIO_INPUT_EN 	&= ~(1<<pinNumber);
	*IOF_EN 		|= (1<<pinNumber);
	*IOF_SEL		|= (1<<pinNumber);
}

void rtc_setup(){
	*RTC_CONFIG &= ~(0b1<<0);
	*RTC_CONFIG &= ~(0b1<<1);
	*RTC_CONFIG &= ~(0b1<<2);

	*RTC_CONFIG |= (0b1<<12);
	printf("\r RTCregister: %u \n", *RTC_CONFIG);
}

uint32_t get_rtc_low(){
	volatile uint32_t rtc_low_value = *RTC_OUTPUT_LOW;
	return rtc_low_value;
}

uint32_t get_rtc_low_micro(){

	volatile uint32_t rtc_low_value = *RTC_OUTPUT_LOW;
	uint64_t rtc_low_value_Multi = rtc_low_value * clock_multiplier_micro;
	uint32_t result = rtc_low_value_Multi / clock_frequency;
	return (result);
}

uint32_t get_rtc_low_milli(){

	volatile uint32_t rtc_low_value = *RTC_OUTPUT_LOW;
	return ((rtc_low_value * clock_multiplier_milli)/clock_frequency);
}

uint32_t get_rtc_high(){
	volatile uint32_t rtc_high_value = *RTC_OUTPUT_HIGH;
	return rtc_high_value;
}

void initSensor() {
	resetSingleShot();
	uint32_t waitTime = get_rtc_low_micro() + 60000;
	while(get_rtc_low_micro() <= waitTime);
	resetSingleShot();
	waitTime = get_rtc_low_micro() + 60000;
	while(get_rtc_low_micro() <= waitTime);
	waitTime = get_rtc_low_micro() + 2000000;
	while(get_rtc_low_micro() <= waitTime);
}

void setDutyCycle1(uint32_t dutyCycle){
	*PWM0_1_CMP1 =((*PWM0_1_CMP0 * (100000 - dutyCycle))/100000);
}

void setPWM1(uint32_t frequency, uint32_t dutyCycle){
	int maxComparator =  256 - 1; // Maximum 8-bit comparator value
	int scale = 0;
	uint32_t comparator = (BASE_FREQUENCY) / (frequency * pow(2,scale));

	while(comparator > maxComparator){
		scale += 1;
		comparator = (BASE_FREQUENCY) / (frequency * pow(2,scale));
		printf("\r PWM1 Comparator: %u\n", comparator);
		printf("\r PWM1 SCALE: %u\n", scale);
	}

	*PWM0_1_CMP0 = 0b0000000000000000;
	*PWM0_1_CMP0 |= comparator;
	*PWM0_1_CONFIG |= scale;
	*PWM0_1_CMP1 = 0b0000000000000000;
	*PWM0_1_CMP1 |= (comparator * ((100000 - dutyCycle))/100000);
	printf("\r PWM1 Comparator0 Final: %u\n", *PWM0_1_CMP0);
	printf("\r PWM1 Comparator1 Final: %u\n", *PWM0_1_CMP1);
	printf("\r PWM1 SCALE FINAL: %u\n", scale);
	printf("\r PWM1 CONFIG FINAL: %u\n", *PWM0_1_CONFIG);
}

void setDutyCycle2(uint32_t dutyCycle){
	*PWM2_1_CMP1 =((*PWM2_1_CMP0 * (100000 - dutyCycle))/100000);
}

void setPWM2(uint32_t frequency, uint32_t dutyCycle){
	int maxComparator =  65536 - 1; // Maximum 16-bit comparator value
	int scale = 0;
	uint32_t comparator = (BASE_FREQUENCY) / (frequency * pow(2,scale));

	while(comparator > maxComparator){
		scale ++;
		comparator = (BASE_FREQUENCY) / (frequency * pow(2,scale));
		printf("\r PWM2 Comparator: %u\n", comparator);
		printf("\r PWM2 SCALE: %u\n", scale);
	}

	*PWM2_1_CMP0 = 0b0000000000000000;
	*PWM2_1_CMP0 |= comparator;
	*PWM2_1_CONFIG |= scale;
	*PWM2_1_CMP1 = 0b0000000000000000;
	*PWM2_1_CMP1 |= ((comparator * (100000 - dutyCycle))/100000);
	printf("\r PWM2 Comparator0 Final: %u\n", *PWM2_1_CMP0);
	printf("\r PWM2 Comparator1 Final: %u\n", *PWM2_1_CMP1);
	printf("\r PWM2 SCALE FINAL: %u\n", scale);
	printf("\r PWM2 CONFIG FINAL: %u\n", *PWM2_1_CONFIG);
}

void setSingleShotPWM(uint32_t pinNumber, uint32_t frequency, uint32_t dutyCycle){
	setPWMPin(pinNumber);
	*PWM0_1_CONFIG = 0b0000000000000000000000000000000;
	*PWM0_1_CONFIG |= (1<<9); 		// ENABLE PWMZeroCMP
	*PWM0_1_CONFIG |= (1<<13); 		// Run PWM01 in ONE-SHOT mode
	printf("\r PWM1 CONFIG FINAL: %u\n", *PWM0_1_CONFIG);
	setPWM1(frequency, dutyCycle);
}

void resetSingleShot() {
	*PWM0_1_CONFIG |= (1<<9); 		// ENABLE PWMZeroCMP
	*PWM0_1_CONFIG |= (1<<13); 		// Run PWM01 in ONE-SHOT mode
}

void setContinuousPWM(uint32_t pinNumber, uint32_t frequency, uint32_t dutyCycle){
	setPWMPin(pinNumber);
	*PWM2_1_CONFIG = 0b0000000000000000000000000000000;
	*PWM2_1_CONFIG |= (1<<9); 		// ENABLE PWMZeroCMP
	*PWM2_1_CONFIG |= (1<<12); 		// Run PWM01 in CONTINUOUS mode
	setPWM2(frequency, dutyCycle);
}

void save_measurement(int distance, int angle, uint32_t measurement){

	// Fill empty array with first measurements
	if(!closestPositions[measurement - 1].distance && distance != 0){
		closestPositions[measurement - 1].distance = distance;
		closestPositions[measurement - 1].angle = angle;
	}
	// Replace position if it is smaller than current (and is not 0 because 0 means error)
	if(distance < closestPositions[measurement - 1].distance && distance != 0){
		closestPositions[measurement - 1].distance = distance;
		closestPositions[measurement - 1].angle = angle;
	}

	for(int i = 0; i<3; i++){
		// Fill empty array with first measurements
		if(!closestMeasurements[i].distance && distance != 0){
			closestMeasurements[i].distance = distance;
			closestMeasurements[i].angle = angle;
			break;
		}
		// Replace position if it is smaller than current (and is not 0 because 0 means error)
		if(distance < closestMeasurements[i].distance && distance != 0){
			closestMeasurements[i].distance = distance;
			closestMeasurements[i].angle = angle;
			break;
		}
	}
}

void streamMeasurements(){
	printf("\r \n");
	printf("\r All the closest measurements: \n");
	for(int i = 0; i < MeasurementAmount; i++){
		printf("\r Measurement: %u \n", i + 1);
		printf("\r Distance: %d \n",closestPositions[i].distance);
		printf("\r Angle: (%u)\n", closestPositions[i].angle);
		printf("\r \n");
	}

}

void stream_output(uint32_t distance, uint32_t angle){
	printf("\r \n");
    printf("\r Results:\n");
    printf("\r Distance: %u\n", distance);
    printf("\r MotorAngle: %u\n", angle);
}

uint32_t read_echo(){
	uint32_t timeout = get_rtc_low_micro() + sensor_timeout;
	while(!((*GPIO_INPUT_VAL >> ECHO_PIN) & 0b1));
	uint32_t startTime = get_rtc_low_micro();
	while(((*GPIO_INPUT_VAL >> ECHO_PIN) & 0b1));
	uint32_t stopTime = get_rtc_low_micro();
	if(get_rtc_low_micro() >= timeout){
		//printf("\r Sensor Timeout: %u \n", get_rtc_low_micro());
		return(0);
	}
	else {
		//printf("\r Got Value\n");
		return(stopTime - startTime);
	}
}

float calcMotorAngle(uint32_t curposition, bool curDirection){
	float positionAngle = 0;
	if(curDirection == false){
		float deltaMeasurement = maxServoAngle / MeasurementAmount;
		positionAngle = deltaMeasurement * curposition;
	}
	else if(curDirection == true){
		float deltaMeasurement = maxServoAngle / MeasurementAmount;
		positionAngle = deltaMeasurement * (MeasurementAmount - curposition);
	}
	return(positionAngle);
}

uint32_t calcNewPosition(uint32_t curposition, bool curDirection){
	uint32_t diffDutyCycle = maxDutyCycle - minDutyCycle;
	uint32_t deltaMeasurement = diffDutyCycle / MeasurementAmount;	// dutycycle change per measurement
	uint32_t newPosition = minDutyCycle;

	if(curDirection == false){
		if(curposition < MeasurementAmount){
			newPosition = (curposition + 1) * deltaMeasurement + minDutyCycle;
		}
		else {
			newPosition = (curposition) * deltaMeasurement + minDutyCycle;
		}
	}

	else if(curDirection == true) {
		if(curposition < MeasurementAmount){
			newPosition = (MeasurementAmount - curposition) * deltaMeasurement + minDutyCycle;
		}
		else {
			newPosition = minDutyCycle;
		}
	}

	return(newPosition);
}
