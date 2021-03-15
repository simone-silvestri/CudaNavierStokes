


#ifndef DISPLAY_H_
#define DISPLAY_H_

#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"
#include <ctime>

#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
using namespace std;

#ifdef LINUX
  #include "GL/gl.h"
  #include "GL/glu.h"
  #include "GL/glut.h"
#else
  #include "/usr/include/GL/gl.h"
  #include "/usr/include/GL/glu.h"
  #include "/usr/include/GL/glut.h"
#endif

#include "globals.h"


int width = 1920;
int height = 800;
float aspectRatio = float(height)/float(width);

int jd = 0;
int kd = 0;

void initDisplay(int argc, char** argv);
void display();
void reshape(GLint w, GLint h);
void key(unsigned char c, int x, int y);



void initDisplay(int argc, char** argv)  {


        if(argc>1) {
	  jd=strtol(argv[1], nullptr, 0);
	}
	if(argc>2) {
          kd=strtol(argv[2], nullptr, 0);	
        }

        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
        glutCreateWindow("Graph plotter");
	glutInitWindowSize(width, height);   			// Set the window's initial width & height - non-square
	glutInitWindowPosition(50, 50); 			// Position the window's initial top-left corner

        /* Register GLUT callbacks. */
        glutDisplayFunc(display);
      	glutKeyboardFunc(key);
	glutReshapeFunc(reshape);
        glutIdleFunc(run);

        /* Init the GL state */
        glLineWidth(1.0);

}


/* Key press processing */
void key(unsigned char c, int x, int y)
{
	if(c == 27) exit(0);
};


void reshape(GLint w, GLint h)
{
	if (h == 0) height = 1;
	aspectRatio = (GLfloat)w / (GLfloat)h;

	// Set the viewport to cover the new window
	glViewport(0, 0, w, h);

	// Set the aspect ratio of the clipping area to match the viewport
	glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
	glLoadIdentity();             // Reset the projection matrix

	gluOrtho2D(-1.0*aspectRatio, 1.0*aspectRatio, -1.0, 1.0);

	width = w;
	height = h;


	glutPostRedisplay();

}



void display() {

        float maxPhi = -1000;
        float minPhi =  1000;
        float xg[mx],phig[mx];
        for (int i=0; i<mx; i++) {
            maxPhi  = MAX(maxPhi,(float)phi[idx(i,jd,kd)]);
            minPhi  = MIN(minPhi,(float)phi[idx(i,jd,kd)]);
            xg[i]   = (float) x[i];
            phig[i] = (float) phi[idx(i,jd,kd)]; } 

	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);

	//glLoadIdentity();

	glPushMatrix();
	glScalef(1.0 / (xg[mx-1]-xg[0]), 1.0 / (maxPhi-minPhi), 1.0);
	glTranslatef(-xg[0], -minPhi, 0.0);
	glColor3f(1.0, 1.0, 1.0);

        glBegin(GL_LINE_STRIP); 
          for(int i=0; i<mx; i++) {
                  glVertex2f(xg[i],phig[i]);
          }
        glEnd();

	glPopMatrix();
	glutSwapBuffers();

	//glFlush();  // Render now
}

#endif







