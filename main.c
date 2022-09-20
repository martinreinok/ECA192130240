#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include "BIT_OPS.h"
#include "RTC.h"
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
void setPWM1(uint32_t frequency, uint32_t dutyCycle);
void setDutyCycle1(uint32_t dutyCycle);
void resetSingleShot();
void setPWM2(uint32_t frequency, uint32_t dutyCycle);
void setDutyCycle2(uint32_t dutyCycle);
void setSingleShotPWM(uint32_t pinNumber, uint32_t frequency, uint32_t dutyCycle);
void setContinuousPWM(uint32_t pinNumber, uint32_t frequency, uint32_t dutyCycle);
uint32_t angleToDutyCycle(uint32_t motorAngle);
void save_measurement(int distance, int angle);
void stream_output(uint32_t distance, uint32_t angle);
void delay_us(uint32_t delay_in_us);
uint32_t read_echo();
uint32_t calcMotorAngle(uint32_t curposition, bool curDirection);
uint32_t calcNewPosition(uint32_t curposition, bool curDirection);


/*Global VARIABLES*/
#define MeasurementAmount 65
#define sweepAmount 3
#define ServoPWMFreq 200
#define minDutyCycle 10000
#define maxDutyCycle 37950
#define maxServoAngle 135
#define triggerDutyCycle 50000
#define triggerPWMFreq 20000
static uint32_t BASE_FREQUENCY = 16000000;
struct Measurement closestPositions [3] = {}; // Closest positions measured
static uint32_t speed_of_sound = 340; // cm/s
static uint32_t sensor_timeout = 25000;  // Units ?
static uint32_t loopTime = 60000;
bool sweepDirection = false;		// Direction of sweep

/*REGISTRY DECLARATIONS*/
volatile uint32_t *GPIO_INPUT_VAL 	= (uint32_t*)0x10012000;		// Registry for INPUT values
volatile uint32_t *GPIO_INPUT_EN 	= (uint32_t*)0x10012004;		// Registry for INPUT set pin ENABLE
volatile uint32_t *GPIO_OUTPUT_VAL 	= (uint32_t*)0x1001200C;		// Registry for OUTPUT values
volatile uint32_t *GPIO_OUTPUT_EN 	= (uint32_t*)0x10012008;		// Registry for OUTPUT set pin ENALBE
volatile uint32_t *GPIO_PUE 		= (uint32_t*)0x10012010;		// Registry for INPUT Internal PULL UP ENABLE
volatile uint32_t *IOF_EN 			= (uint32_t*)0x10012038;		// Registry for HW-Driven funtions (IOF)
volatile uint32_t *IOF_SEL 			= (uint32_t*)0x1001203C;		// Registry for HW-Driven funtions (IOF)

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
	setContinuousPWM(SERVO_PIN, ServoPWMFreq, minDutyCycle);
	rtc_setup();

	while(1) {
		uint32_t sweep;

		for(sweep = 1; sweep <- sweepAmount; sweep ++){
			uint32_t measurement;

			for(measurement = 0; measurement <= MeasurementAmount; measurement ++){
				uint32_t motor_angle = calcMotorAngle(measurement + 1, sweepDirection);

				uint32_t distance = 0;

				// Take an average of 3 measurements
				uint32_t i;
				for(i = 1; i <= 3; i ++){
					uint32_t minLoopTime = get_rtc_low_micro() + loopTime;	// Minimum time for loop to complete
					// Trigger sensor
					resetSingleShot();
					// Measure the distance;
					distance += (read_echo() * speed_of_sound)/2000;

					while(get_rtc_low_micro() <= minLoopTime); 			// wait for minimum looptime
				}

				distance = distance / 3;
				if(distance == 0){
					distance = 3000;
				}

				/*
				// Trigger sensor
				resetSingleShot();
				// Measure the distance;
				uint32_t distance = (read_echo() * speed_of_sound)/2000;
				*/
				printf("\r \n");
				printf("\r Current measurement: %u \n", measurement);
				
				// Print results in terminal
				stream_output(distance, motor_angle);
				save_measurement(distance, motor_angle);
		        printf("\r First: %d (%d), Second: %d (%d), Third: %d (%d) \n", closestPositions[0].distance, closestPositions[0].angle, closestPositions[1].distance, closestPositions[1].angle, closestPositions[2].distance, closestPositions[2].angle);

				setDutyCycle2(calcNewPosition(measurement, sweepDirection));
			}
			sweepDirection = !sweepDirection;
		}

		// Print results in terminal
		// stream_output(int distance, int angle);

		// Move motor to next position here

		// If measurements are over


		/*
		Print all array distances:
		for(int i=0; i<array_element_index; i++){
            	printf("Array distance[%d]: %d\n", i, measurements[i].distance);
        	}
		Sort the array:
		bubble_sort(measurements, array_element_index);
		*/
	}
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

void setDutyCycle1(uint32_t dutyCycle){
	*PWM0_1_CMP1 =((*PWM0_1_CMP0 * (100000 - dutyCycle))/100000);
}

void setPWM1(uint32_t frequency, uint32_t dutyCycle){
	int maxComparator =  256 - 1; // Maximum 16-bit comparator value
	//int scale = 0;
	uint32_t comparator; //= BASE_FREQUENCY / (frequency * pow(2,scale));

	int scale = -1;

	do
	{
		scale++;
		comparator = (BASE_FREQUENCY) / (frequency * pow(2,scale));
		printf("\r PWM1 Comparator: %u\n", comparator);
		printf("\r PWM1 SCALE: %u\n", scale);
	} while (comparator > maxComparator);
	/*
	while(comparator > maxComparator){
		scale += 1;
		comparator = BASE_FREQUENCY / (frequency * pow(2,scale));
		if(comparator <= maxComparator) {
			scale -= 1;
			printf("\r SCALE Break: %u\n", scale);
			break;
		}
		printf("\r SCALE: %u\n", scale);
	}
	*/
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
	//int scale = 0;
	uint32_t comparator; //= BASE_FREQUENCY / (frequency * pow(2,scale));

	int scale = -1;

	do
	{
		scale++;
		comparator = (BASE_FREQUENCY) / (frequency * pow(2,scale));
		printf("\r PWM2 Comparator: %u\n", comparator);
		printf("\r PWM2 SCALE: %u\n", scale);
	} while (comparator > maxComparator);
	/*
	while(comparator > maxComparator){
		scale += 1;
		comparator = BASE_FREQUENCY / (frequency * pow(2,scale));
		if(comparator <= maxComparator) {
			scale -= 1;
			printf("\r SCALE Break: %u\n", scale);
			break;
		}
		printf("\r SCALE: %u\n", scale);
	}
	*/
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

void save_measurement(int distance, int angle){
	for(int i = 0; i<3; i++){

		// Fill empty array with first measurements
		if(!closestPositions[i].distance){
			closestPositions[i].distance = distance;
			closestPositions[i].angle = angle;
			break;
		}
		// Replace position if it is smaller than current (and is not 0 because 0 means error)
		if(distance < closestPositions[i].distance && distance != 0){
			closestPositions[i].distance = distance;
			closestPositions[i].angle = angle;
			break;
		}
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

uint32_t calcMotorAngle(uint32_t curposition, bool curDirection){
	if(curDirection == false){
		uint32_t deltaMeasurement = maxServoAngle / MeasurementAmount;
		uint32_t positionAngle = deltaMeasurement * curposition;
		return(positionAngle);
	}
	else if(curDirection == true){
		uint32_t deltaMeasurement = maxServoAngle / MeasurementAmount;
		uint32_t positionAngle = deltaMeasurement * (MeasurementAmount - curposition);
		return(positionAngle);
	}
}

uint32_t calcNewPosition(uint32_t curposition, bool curDirection){
	uint32_t diffDutyCycle = maxDutyCycle - minDutyCycle;
	uint32_t deltaMeasurement = diffDutyCycle / MeasurementAmount;	// dutycycle change per measurement
	uint32_t newPosition = minDutyCycle;

	if(curDirection == false){
		if(curposition < MeasurementAmount){
			newPosition = (curposition + 1) * deltaMeasurement + minDutyCycle;
			return(newPosition);
		}
		else {
			newPosition = (curposition) * deltaMeasurement + minDutyCycle;
			return(newPosition);
		}
	}

	else if(curDirection == true) {
		if(curposition < MeasurementAmount){
			newPosition = (MeasurementAmount - curposition) * deltaMeasurement + minDutyCycle;
			return(newPosition);
		}
		else {
			newPosition = minDutyCycle;
			return(newPosition);
		}
	}
}
