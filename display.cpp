#include <SFML/Graphics.hpp>


void drawGrid()
{
    int columns = 3;
    int rows = 3;

    sf::RenderWindow window(sf::VideoMode(800, 600), "TicTacToe");
    sf::RectangleShape grid[columns][rows];

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        sf::Vector2f cellSize(200.0f, 200.0f);

        for (int i = 0; i < columns; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                grid[i][j].setSize(cellSize);
                grid[i][j].setOutlineColor(sf::Color::Blue);
                grid[i][j].setOutlineThickness(5.0f);

                grid[i][j].setPosition(i * cellSize.x + 5.0f, j * cellSize.y + 5.0f);

                window.draw(grid[i][j]);
            }
        }
        window.display();
    }
}

/*float xdim = 800.0 / gx;
float ydim = 600.0 / gy;

set_draw_mode(DRAW_NORMAL); // Should set this if your program does any XOR drawing in callbacks.
clearscreen();              /* Should precede drawing for all drawscreens 

setfontsize(10);
setlinestyle(SOLID);
setlinewidth(5);
setcolor(BLACK);

//Grid outline
drawrect(50, 50, 850, 650);
int x1, x2, y1, y2;
int stepx = 30;
int stepy = 35;

//Wire indicator boxes
for (int i = 0; i < nw; i++)
{
    setcolor(W[i].colour);
    fillrect(50 + (stepx), 700 + (i * stepy), (50 + (stepx) + 25), (700 + (i * stepy)) + 25);
}

//Grid lines
setcolor(BLACK);
setlinewidth(2);
for (int i = 0; i < gx; i++)
{
    y1 = 50;
    y2 = 650;
    x1 = (i * xdim) + 50;
    x2 = x1;
    drawline(x1, y1, x2, y2);
}
for (int i = 0; i < gy; i++)
{
    x1 = 50;
    x2 = 850;
    y1 = (i * ydim) + 50;
    y2 = y1;
    drawline(x1, y1, x2, y2);
}

//Cells
int xx, yy;
setcolor(DARKGREY);
for (int i = 0; i < nc; i++)
{
    xx = cells[i][0];
    yy = cells[i][1];
    fillrect((xdim * xx) + 50, (ydim * yy) + 50, (xdim * xx) + xdim + 50, (ydim * yy) + ydim + 50);
}

//Pins
int np;
for (int i = 0; i < nw; i++)
{
    np = W[i].numPins;
    for (int j = 0; j < np; j++)
    {
        xx = W[i].pins[j][0];
        yy = W[i].pins[j][1];
        setcolor(i + 4);
        setlinestyle(SOLID);
        setlinewidth(2);
        fillrect((xdim * xx) + 50, (ydim * yy) + 50, (xdim * xx) + xdim + 50, (ydim * yy) + ydim + 50);

        //Highlight the source
        if (j == 0)
        {
            setcolor(BLACK);
            setlinestyle(DASHED);
            setlinewidth(2);
            drawrect((xdim * xx) + 50, (ydim * yy) + 50, (xdim * xx) + xdim + 50, (ydim * yy) + ydim + 50);
        }
    }
}*/
