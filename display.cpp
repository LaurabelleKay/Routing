#include "display.h"


using namespace std;

void drawGrid(int gridx, int gridy, Point **points, Wire *W)
{
    sf::RenderWindow window(sf::VideoMode(800, 600), "Grid");
    vector<vector<sf::RectangleShape>> grid(gridx);


    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        sf::Vector2f cellSize((float)800.0/gridx, (float)600.0/gridy);

        for (int i = 0; i < gridx; i++)
        {
            grid[i] = vector<sf::RectangleShape>(gridy);
            for (int j = 0; j < gridy; j++)
            {
                grid[i][j].setSize(cellSize);
                grid[i][j].setFillColor(sf::Color::White);
                if(points[i][j].obstructedBy == -1)
                {
                    grid[i][j].setFillColor(sf::Color::Black);
                }
                else if(points[i][j].obstructedBy != -2)
                {
                    int ind = points[i][j].obstructedBy;
                    grid[i][j].setFillColor(sf::Color(W[ind].r, W[ind].g, W[ind].b));
                }
                grid[i][j].setOutlineColor(sf::Color::Black);
                grid[i][j].setOutlineThickness(1.0f);

                grid[i][j].setPosition(i * cellSize.x + 5.0f, j * cellSize.y + 5.0f);

                window.draw(grid[i][j]);
            }
        }
        window.display();
    }
}
